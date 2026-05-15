import copy
import datetime
import gc
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods._trainer import _Trainer
from utils.router_eval import (
    build_router_quality_summary,
    compute_oracle_accuracies,
    compute_router_metrics,
)

logger = logging.getLogger()


class MVP(_Trainer):
    def __init__(self, **kwargs):
        super(MVP, self).__init__(**kwargs)

        self.use_afs  = True
        self.use_mcr  = True
        self.use_mask = True

        self.alpha  = 0.5
        self.gamma  = 2.0
        self.margin  = 0.5

        self.task_id = 0
    @torch.no_grad()
    def _collect_rp_features(self, images, labels):
        """Collect features for RPFC gating (if enabled).

        Uses backbone CLS features and current internal step id as regression
        targets, following the FlyPrompt RPFC design.
        """
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        rp_head = getattr(self.model_without_ddp, "rp_head", None)
        if not use_rp_gate or rp_head is None:
            return

        # Map labels to seen-class indices, consistent with training.
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device)

        images = self.test_transform_tensor(images)

        self.model_without_ddp.backbone.eval()
        if hasattr(self.model_without_ddp.backbone, "forward_features"):
            feats = self.model_without_ddp.backbone.forward_features(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            cls_feat = feats[:, 0]
        else:
            x = self.model_without_ddp.backbone.patch_embed(images)
            B, N, D = x.size()
            cls_token = self.model_without_ddp.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.model_without_ddp.backbone.pos_drop(token_appended + self.model_without_ddp.backbone.pos_embed)
            x = self.model_without_ddp.backbone.blocks(x)
            x = self.model_without_ddp.backbone.norm(x)
            cls_feat = x[:, 0]

        step_id = getattr(self, "current_step", 0)
        step_labels = torch.full(
            (labels.size(0),),
            step_id,
            device=labels.device,
            dtype=torch.long,
        )
        self.model_without_ddp.rp_head.collect(cls_feat, step_labels)



    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        # Collect RPFC features once per online_step (per internal step), using
        # the original images/labels (before deletion).
        self._collect_rp_features(images.clone(), labels.clone())

        # Update internal step schedule based only on the number of samples
        # seen (task-boundary-free).
        if hasattr(self, "_maybe_advance_internal_step"):
            batch_size_global = images.size(0) * self.world_size
            self._maybe_advance_internal_step(batch_size_global)

        del (images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x,y,mask=logit_mask)
        else:
            logit, loss = self.model_forward(x,y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        # EMA classifier head update (optional, per-prompt-slot experts)
        if getattr(self.model_without_ddp, "use_ema_head", False) and hasattr(self.model_without_ddp, "update_ema_fc"):
            expert_ids = getattr(self.model_without_ddp, "last_expert_ids", None)
            if expert_ids is not None:
                self.model_without_ddp.update_ema_fc(expert_ids=expert_ids)


        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            feature, mvp_mask = self.model_without_ddp.forward_features(x)
            logit = self.model_without_ddp.forward_head(feature)
            if mask is not None: # batchmask
                logit += mask
            elif self.use_mask:
                logit = logit * mvp_mask
                logit = logit + self.mask
            loss = self.loss_fn(feature, mvp_mask, y)
        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        # If RPFC gating is enabled, update RPFC weights before evaluation.
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        rp_head = getattr(self.model_without_ddp, "rp_head", None)
        if use_rp_gate and rp_head is not None:
            self.model_without_ddp.rp_head.update()

        self.model.eval()
        router_mode = getattr(self, "router_mode", "learned")
        log_router_trace = getattr(self, "log_router_trace", False)
        analysis_router_quality = getattr(self, "analysis_router_quality", False)
        oracle_modes = ["sample_oracle", "class_oracle", "worst"]
        final_eval = end and (task_id is None or task_id == self.n_tasks - 1)

        results_summary = {}
        if router_mode in oracle_modes or (analysis_router_quality and final_eval):
            results_summary = self._collect_mvp_oracle_results(test_loader)

        eval_mode = router_mode if router_mode in ["learned", "random", "single"] else "learned"
        eval_stats = self._evaluate_mvp_mode(
            test_loader,
            mode=eval_mode,
            collect_trace=log_router_trace,
            task_id=task_id,
        )

        avg_acc = eval_stats["avg_acc"]
        if router_mode == "sample_oracle":
            avg_acc = results_summary.get("A_sample_oracle", avg_acc)
        elif router_mode == "class_oracle":
            avg_acc = results_summary.get("A_class_oracle", avg_acc)
        elif router_mode == "worst":
            avg_acc = results_summary.get("A_worst", avg_acc)

        if log_router_trace and self.is_main_process():
            os.makedirs(self.log_dir, exist_ok=True)
            trace_path = os.path.join(self.log_dir, f"router_trace_seed_{self.rnd_seed}.pt")
            torch.save(eval_stats["traces"], trace_path)
            logger.info("[MVP] Saved router trace to %s", trace_path)

        if analysis_router_quality and self.is_main_process() and final_eval:
            learned_stats = self._evaluate_mvp_mode(test_loader, mode="learned", collect_trace=True, task_id=task_id)
            random_stats = self._evaluate_mvp_mode(test_loader, mode="random", collect_trace=False, task_id=task_id)
            single_stats = self._evaluate_mvp_mode(test_loader, mode="single", collect_trace=False, task_id=task_id)

            learned_traces = learned_stats["traces"]
            metrics = compute_router_metrics(
                [t["label"] for t in learned_traces],
                [t["route_id"] for t in learned_traces],
                [t["route_margin"] for t in learned_traces],
                num_routes=self._mvp_num_routes(),
            )
            summary = build_router_quality_summary(
                method="mvp",
                dataset=self.dataset,
                seed=self.rnd_seed,
                A_learned=learned_stats["avg_acc"],
                A_random=random_stats["avg_acc"],
                A_single=single_stats["avg_acc"],
                A_class_oracle=results_summary.get("A_class_oracle"),
                A_sample_oracle=results_summary.get("A_sample_oracle"),
                A_worst=results_summary.get("A_worst"),
                metrics=metrics,
            )
            summary_path = os.path.join(self.log_dir, f"router_quality_summary_seed_{self.rnd_seed}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("[MVP] Saved router quality summary to %s", summary_path)

        eval_dict = {
            "avg_loss": eval_stats["avg_loss"],
            "avg_acc": avg_acc,
            "cls_acc": eval_stats["cls_acc"],
        }
        return eval_dict

    def _mvp_num_routes(self):
        return int(getattr(self.model_without_ddp, "e_pool", getattr(self.model_without_ddp, "task_num", self.n_tasks)))

    def _mvp_forward_logits(self, x, mode="learned", forced_route_ids=None):
        use_ema_head = getattr(self.model_without_ddp, "use_ema_head", False)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if use_ema_head:
                logit_ls = self.model_without_ddp.forward_with_ema(
                    x,
                    expert_ids=forced_route_ids,
                    router_mode=mode,
                )
                if getattr(self, "eval_ema_only", False):
                    logit_ls = logit_ls[1:]
                logit_ls = [logit + self.mask for logit in logit_ls]
                logit = self._ensemble_logits(logit_ls)
            else:
                feature, mvp_mask = self.model_without_ddp.forward_features(
                    x,
                    forced_route_ids=forced_route_ids,
                    router_mode=mode,
                )
                logit = self.model_without_ddp.forward_head(feature)
                if self.use_mask:
                    logit = logit * mvp_mask
                logit = logit + self.mask
        return logit

    def _evaluate_mvp_mode(self, test_loader, mode="learned", collect_trace=False, task_id=None):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        router_traces = []
        sample_offset = 0

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self._mvp_forward_logits(x, mode=mode)
                loss = F.cross_entropy(logit, y)
                sample_losses = F.cross_entropy(logit, y, reduction="none")

                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
                if collect_trace:
                    route_ids = getattr(self.model_without_ddp, "last_route_ids", None)
                    route_topk = getattr(self.model_without_ddp, "last_route_topk", None)
                    score_matrix = getattr(self.model_without_ddp, "last_route_score_matrix", None)
                    learned_topk = getattr(self.model_without_ddp, "last_learned_route_topk", None)
                    margins = getattr(self.model_without_ddp, "last_route_margin", None)

                    if route_ids is not None:
                        route_ids = route_ids.detach().cpu()
                        route_topk = route_topk.detach().cpu() if route_topk is not None else route_ids.unsqueeze(1)
                        score_matrix = score_matrix.detach().cpu() if score_matrix is not None else None
                        learned_topk = learned_topk.detach().cpu() if learned_topk is not None else route_topk
                        for bi in range(x.size(0)):
                            route_id = int(route_ids[bi].item())
                            if score_matrix is None:
                                route_scores = []
                            else:
                                route_scores = [float(v) for v in score_matrix[bi].tolist()]
                            route_margin = float(margins[bi]) if margins is not None else 0.0
                            router_traces.append({
                                "sample_id": int(sample_offset + bi),
                                "label": int(y[bi].item()),
                                "pred": int(pred[bi].item()),
                                "correct": bool(pred[bi].item() == int(y[bi].item())),
                                "loss": float(sample_losses[bi].detach().cpu().item()),
                                "method": getattr(self, "method", "mvp"),
                                "dataset": getattr(self, "dataset", "cifar100"),
                                "seed": int(self.rnd_seed),
                                "eval_step": int(task_id) if task_id is not None else None,
                                "batch_id": int(i),
                                "router_mode": mode,
                                "route_id": route_id,
                                "route_topk": [int(v) for v in route_topk[bi].tolist()],
                                "learned_route_topk": [int(v) for v in learned_topk[bi].tolist()],
                                "route_scores": route_scores,
                                "score_direction": "higher_is_better",
                                "route_margin": route_margin,
                            })
                sample_offset += x.size(0)

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        return {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "cls_acc": cls_acc,
            "traces": router_traces,
        }

    def _collect_mvp_oracle_results(self, test_loader):
        n_routes = self._mvp_num_routes()
        max_routes = getattr(self, "oracle_max_routes", -1)
        if max_routes > 0:
            n_routes = min(n_routes, max_routes)

        per_route_logits = [[] for _ in range(n_routes)]
        all_labels = []
        oracle_batch_size = getattr(self, "oracle_batch_size", -1)

        with torch.no_grad():
            for data in test_loader:
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)
                chunk_size = x.size(0) if oracle_batch_size <= 0 else oracle_batch_size

                for start in range(0, x.size(0), chunk_size):
                    x_sub = x[start:start + chunk_size]
                    y_sub = y[start:start + chunk_size]
                    batch_size = x_sub.size(0)
                    for r in range(n_routes):
                        route_ids = torch.full((batch_size,), r, dtype=torch.long, device=self.device)
                        logit = self._mvp_forward_logits(x_sub, forced_route_ids=route_ids)
                        per_route_logits[r].append(logit.detach().cpu())
                    all_labels.append(y_sub.detach().cpu())

        per_route_logits = [torch.cat(parts, dim=0) for parts in per_route_logits]
        all_labels = torch.cat(all_labels, dim=0)
        return compute_oracle_accuracies(per_route_logits, all_labels, n_classes=self.n_classes)

    def _ensemble_logits(self, logit_ls):
        """Ensemble a list of logits from online and EMA heads.

        The behavior mirrors FlyPrompt's implementation and is controlled by
        ``self.ensemble_method`` (default: softmax_max_prob).
        """
        if not hasattr(self, "ensemble_method"):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # [batch_size, n_classes, n_heads]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)
            min_entropy_indices = torch.argmin(entropies, dim=-1)
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    @torch.no_grad()
    def analyze_expert_features(self):
        """Post-hoc expert feature analysis for MVP.

        After all tasks have been trained and evaluated, this method:
        1) Extracts CLS features for every expert (task) on *all* test samples
           using the fixed backbone and task-specific prompts.
        2) Saves the tensor of features with shape [n_tasks, n_samples, dim].
        3) Computes pairwise cosine similarity between per-task mean features.
        4) Computes pairwise linear CKA between task representations.

        The results are saved under ``self.log_dir`` with filenames
        ``features_seed_*.pt``, ``similarity_seed_*.npy`` and
        ``cka_seed_*.npy``, plus corresponding heatmap figures if
        matplotlib is available.
        """
        if not hasattr(self, "model_without_ddp"):
            logger.warning("[MVP] model_without_ddp not found, skip expert analysis.")
            return
        if not hasattr(self, "test_dataset"):
            logger.warning("[MVP] test_dataset not found, skip expert analysis.")
            return

        model = self.model_without_ddp
        model.eval()
        if hasattr(model, "backbone"):
            model.backbone.eval()

        device = self.device
        # Number of experts in the model may differ from benchmark n_tasks
        # when using internal step-based scheduling.
        n_experts = getattr(model, "task_num", self.n_tasks)

        # Build deterministic DataLoader over the full test set
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batchsize * 2,
            shuffle=False,
            num_workers=self.n_worker,
            pin_memory=True,
        )

        # Feature dimension equals backbone embedding dimension (CLS token)
        feat_dim = getattr(model.backbone, "num_features", model.backbone.embed_dim)
        num_samples = len(self.test_dataset)
        features = torch.zeros(n_experts, num_samples, feat_dim, dtype=torch.float32)
        common_features = torch.zeros(num_samples, feat_dim, dtype=torch.float32)

        logger.info(
            f"[MVP] Extracting CLS features for {n_experts} experts over "
            f"{num_samples} test samples (dim={feat_dim}) ..."
        )

        offset = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device, non_blocking=True)
                batch_size = images.size(0)

                # Shared backbone tokens (without prompts)
                x = model.backbone.patch_embed(images)
                B, N, D = x.size()
                cls_token = model.backbone.cls_token.expand(B, -1, -1)
                token_appended = torch.cat((cls_token, x), dim=1)
                x_tokens = model.backbone.pos_drop(token_appended + model.backbone.pos_embed)

                # Global prompts (g_prompts) are shared across tasks
                g_prompts = model.g_prompts[0].repeat(B, 1, 1)

                idx_slice = slice(offset, offset + batch_size)

                # Common representation: frozen backbone + shared global prompts, with
                # expert-specific prompts zeroed out.
                base_e_prompt = model.e_prompts[0].unsqueeze(0).repeat(B, 1, 1)
                e_prompt_zero = torch.zeros_like(base_e_prompt)
                x_common = model.prompt_func(x_tokens, g_prompts, e_prompt_zero)
                x_common = model.backbone.norm(x_common)
                common_cls = x_common[:, 0]
                common_features[idx_slice, :] = common_cls.detach().cpu()

                # For each expert t, use its dedicated e_prompt slot (since e_pool == task_num)
                for t in range(n_experts):
                    e_prompt_t = model.e_prompts[t].unsqueeze(0).repeat(B, 1, 1)
                    x_prom = model.prompt_func(x_tokens, g_prompts, e_prompt_t)
                    x_prom = model.backbone.norm(x_prom)
                    cls_feat = x_prom[:, 0]

                    features[t, idx_slice, :] = cls_feat.detach().cpu()

                offset += batch_size

        # ---------- Save raw features ----------
        os.makedirs(self.log_dir, exist_ok=True)
        feat_path = os.path.join(self.log_dir, f"features_seed_{self.rnd_seed}.pt")
        torch.save(features, feat_path)
        logger.info(f"[MVP] Saved expert features to {feat_path}")

        # Also save the common (backbone + shared prompt) features used for residual analysis
        common_feat_path = os.path.join(
            self.log_dir, f"common_features_seed_{self.rnd_seed}.pt"
        )
        torch.save(common_features, common_feat_path)
        logger.info(
            f"[MVP] Saved common (backbone+g_prompts) features to {common_feat_path}"
        )

        # ---------- Cosine similarity between per-task mean features ----------
        mean_feats = features.mean(dim=1)  # [T, D]
        mean_norm = mean_feats / (mean_feats.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = mean_norm @ mean_norm.t()  # [T, T]

        sim_path = os.path.join(self.log_dir, f"similarity_seed_{self.rnd_seed}.npy")
        np.save(sim_path, sim_matrix.numpy())
        logger.info(f"[MVP] Saved expert similarity matrix to {sim_path}")

        # Plot heatmap for similarity matrix if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("MVP expert similarity (cosine of mean CLS)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            sim_fig_path = os.path.join(
                self.log_dir, f"similarity_seed_{self.rnd_seed}.png"
            )
            fig.savefig(sim_fig_path)
            plt.close(fig)
            logger.info(f"[MVP] Saved similarity heatmap to {sim_fig_path}")
        except Exception as e:
            logger.exception(
                "[MVP] Failed to plot similarity heatmap: %s", e
            )

        # ---------- Linear CKA between expert representations ----------
        def _center_gram(x: torch.Tensor) -> torch.Tensor:
            # x: [N, D]
            n = x.size(0)
            unit = torch.ones((n, n), device=x.device)
            identity = torch.eye(n, device=x.device)
            h = identity - unit / n
            k = x @ x.t()
            return h @ k @ h

        def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
            x = x - x.mean(0, keepdim=True)
            y = y - y.mean(0, keepdim=True)

            kx = _center_gram(x)
            ky = _center_gram(y)

            hsic = (kx * ky).sum()
            norm_x = torch.sqrt((kx * kx).sum() + 1e-8)
            norm_y = torch.sqrt((ky * ky).sum() + 1e-8)
            return (hsic / (norm_x * norm_y)).item()

        cka_matrix = torch.zeros(n_experts, n_experts, dtype=torch.float32)
        for i in range(n_experts):
            x_i = features[i]  # [N, D]
            for j in range(n_experts):
                y_j = features[j]
                cka_matrix[i, j] = _cka(x_i, y_j)

        cka_path = os.path.join(self.log_dir, f"cka_seed_{self.rnd_seed}.npy")
        np.save(cka_path, cka_matrix.numpy())
        logger.info(f"[MVP] Saved expert CKA matrix to {cka_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("MVP expert CKA")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            cka_fig_path = os.path.join(self.log_dir, f"cka_seed_{self.rnd_seed}.png")
            fig.savefig(cka_fig_path)
            plt.close(fig)
            logger.info(f"[MVP] Saved CKA heatmap to {cka_fig_path}")
        except Exception as e:
            logger.exception("[MVP] Failed to plot CKA heatmap: %s", e)

        # ---------- Residual expert analysis: (expert - mean over experts) ----------
        # Ref shape: [1, N, D], residual shape: [T, N, D]
        ref = features.mean(dim=0, keepdim=True)
        residual = features - ref

        residual_feat_path = os.path.join(
            self.log_dir, f"residual_features_seed_{self.rnd_seed}.pt"
        )
        torch.save(residual, residual_feat_path)
        logger.info(
            f"[MVP] Saved residual expert features to {residual_feat_path}"
        )

        # Cosine similarity between per-task mean residual features
        residual_mean = residual.mean(dim=1)  # [T, D]
        residual_mean_norm = residual_mean / (
            residual_mean.norm(dim=1, keepdim=True) + 1e-8
        )
        residual_sim_matrix = residual_mean_norm @ residual_mean_norm.t()

        residual_sim_path = os.path.join(
            self.log_dir, f"residual_similarity_seed_{self.rnd_seed}.npy"
        )
        np.save(residual_sim_path, residual_sim_matrix.numpy())
        logger.info(
            f"[MVP] Saved residual expert similarity matrix to {residual_sim_path}"
        )

        # Plot heatmap for residual similarity matrix if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(residual_sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("MVP residual expert similarity (cosine of mean CLS)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            residual_sim_fig_path = os.path.join(
                self.log_dir, f"residual_similarity_seed_{self.rnd_seed}.png"
            )
            fig.savefig(residual_sim_fig_path)
            plt.close(fig)
            logger.info(
                f"[MVP] Saved residual similarity heatmap to {residual_sim_fig_path}"
            )
        except Exception as e:
            logger.exception(
                "[MVP] Failed to plot residual similarity heatmap: %s", e
            )

        # Linear CKA between residual expert representations
        residual_cka_matrix = torch.zeros(n_experts, n_experts, dtype=torch.float32)
        for i in range(n_experts):
            x_i = residual[i]
            for j in range(n_experts):
                y_j = residual[j]
                residual_cka_matrix[i, j] = _cka(x_i, y_j)

        residual_cka_path = os.path.join(
            self.log_dir, f"residual_cka_seed_{self.rnd_seed}.npy"
        )
        np.save(residual_cka_path, residual_cka_matrix.numpy())
        logger.info(
            f"[MVP] Saved residual expert CKA matrix to {residual_cka_path}"
        )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(residual_cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("MVP residual expert CKA")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            residual_cka_fig_path = os.path.join(
                self.log_dir, f"residual_cka_seed_{self.rnd_seed}.png"
            )
            fig.savefig(residual_cka_fig_path)
            plt.close(fig)
            logger.info(
                f"[MVP] Saved residual CKA heatmap to {residual_cka_fig_path}"
            )
        except Exception as e:
            logger.exception("[MVP] Failed to plot residual CKA heatmap: %s", e)



    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        """Hook called after each benchmark task.

        We keep ``task_id`` for logging/analysis only; the underlying model's
        internal step state is advanced exclusively via the task-free
        ``_maybe_advance_internal_step`` scheduler.
        """
        self.task_id += 1

    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.model_without_ddp.backbone.fc)
        head.zero_grad()
        logit = head(feature.detach())
        if self.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask

        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim

        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim

        return sample_grad, batch_grad

    def _get_ignore(self, sample_grad, batch_grad):
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1))#B
        return ign_score

    def _get_compensation(self, y, feat):
        head_w = self.model_without_ddp.backbone.fc.weight[y].clone().detach()
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score

    def loss_fn(self, feature, mask, y):
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)

        if self.use_afs:
            logit = self.model_without_ddp.forward_head(feature)
            logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.model_without_ddp.forward_head(feature)
        if self.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(log_p, y)

        if self.use_mcr:
            loss = (1-self.alpha)* loss + self.alpha * (ign_score ** self.gamma) * loss
        return loss.mean() + self.model_without_ddp.get_similarity_loss()

    def report_training(self, sample_num, train_loss, train_acc):
        logger.info(
             f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
             f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
             f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
             f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
             f"N_Prompts {self.model_without_ddp.e_prompts.size(0)} | "
             f"N_Exposed {len(self.exposed_classes)} | "
             f"Counts {self.model_without_ddp.count.to(torch.int64).tolist()}"
             )
