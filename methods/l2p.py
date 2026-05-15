import gc
import json
import logging
import os

import torch
import torch.nn.functional as F

from methods._trainer import _Trainer
from utils.router_eval import build_router_quality_summary, compute_router_metrics

logger = logging.getLogger()


class L2P(_Trainer):
    def __init__(self, *args, **kwargs):
        super(L2P, self).__init__(*args, **kwargs)

        self.task_id = 0

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        del(images, labels)
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

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        router_mode = getattr(self, "router_mode", "learned")
        if router_mode in {"sample_oracle", "class_oracle", "worst"}:
            raise NotImplementedError("L2P route oracle over top-k prompt combinations is not implemented")

        final_eval = end and (task_id is None or task_id == self.n_tasks - 1)
        eval_stats = self._evaluate_l2p_mode(
            test_loader,
            mode=router_mode if router_mode in {"learned", "random", "single"} else "learned",
            collect_trace=getattr(self, "log_router_trace", False),
            task_id=task_id,
        )

        if getattr(self, "log_router_trace", False) and self.is_main_process():
            os.makedirs(self.log_dir, exist_ok=True)
            trace_path = os.path.join(self.log_dir, f"router_trace_seed_{self.rnd_seed}.pt")
            torch.save(eval_stats["traces"], trace_path)
            logger.info("[L2P] Saved router trace to %s", trace_path)

        if getattr(self, "analysis_router_quality", False) and self.is_main_process() and final_eval:
            learned_stats = self._evaluate_l2p_mode(test_loader, mode="learned", collect_trace=True, task_id=task_id)
            random_stats = self._evaluate_l2p_mode(test_loader, mode="random", collect_trace=False, task_id=task_id)
            single_stats = self._evaluate_l2p_mode(test_loader, mode="single", collect_trace=False, task_id=task_id)
            learned_traces = learned_stats["traces"]
            metrics = compute_router_metrics(
                [t["label"] for t in learned_traces],
                [t["route_id"] for t in learned_traces],
                [t["route_margin"] for t in learned_traces],
                num_routes=getattr(self.model_without_ddp.prompt, "pool_size", self.n_tasks),
            )
            summary = build_router_quality_summary(
                method="l2p",
                dataset=self.dataset,
                seed=self.rnd_seed,
                A_learned=learned_stats["avg_acc"],
                A_random=random_stats["avg_acc"],
                A_single=single_stats["avg_acc"],
                metrics=metrics,
            )
            summary_path = os.path.join(self.log_dir, f"router_quality_summary_seed_{self.rnd_seed}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("[L2P] Saved router quality summary to %s", summary_path)

        return {
            "avg_loss": eval_stats["avg_loss"],
            "avg_acc": eval_stats["avg_acc"],
            "cls_acc": eval_stats["cls_acc"],
        }

    def _evaluate_l2p_mode(self, test_loader, mode="learned", collect_trace=False, task_id=None):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        router_traces = []
        sample_offset = 0

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x, router_mode=mode)
                logit = logit + self.mask
                loss = self.criterion(logit, y)
                sample_losses = F.cross_entropy(logit, y, reduction="none")
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                if collect_trace:
                    selected = getattr(self.model_without_ddp.prompt, "last_selected_indices", None)
                    score_matrix = getattr(self.model_without_ddp.prompt, "last_route_score_matrix", None)
                    margins = getattr(self.model_without_ddp.prompt, "last_route_margin", None)
                    if selected is not None:
                        selected = selected.detach().cpu()
                        score_matrix = score_matrix.detach().cpu() if score_matrix is not None else None
                        margins = margins.detach().cpu() if margins is not None else None
                        learned_topk = None
                        if score_matrix is not None:
                            _, learned_topk = torch.topk(score_matrix, k=min(selected.size(1), score_matrix.size(1)), dim=1)
                        for bi in range(x.size(0)):
                            route_topk = [int(v) for v in selected[bi].tolist()]
                            route_scores = [] if score_matrix is None else [float(v) for v in score_matrix[bi].tolist()]
                            router_traces.append({
                                "sample_id": int(sample_offset + bi),
                                "label": int(y[bi].item()),
                                "pred": int(pred[bi].item()),
                                "correct": bool(pred[bi].item() == int(y[bi].item())),
                                "loss": float(sample_losses[bi].detach().cpu().item()),
                                "method": getattr(self, "method", "l2p"),
                                "dataset": getattr(self, "dataset", "cifar100"),
                                "seed": int(self.rnd_seed),
                                "eval_step": int(task_id) if task_id is not None else None,
                                "batch_id": int(i),
                                "router_mode": mode,
                                "route_id": int(route_topk[0]),
                                "route_topk": route_topk,
                                "learned_route_topk": (
                                    [int(v) for v in learned_topk[bi].tolist()]
                                    if learned_topk is not None else route_topk
                                ),
                                "route_scores": route_scores,
                                "score_direction": "higher_is_better",
                                "route_margin": float(margins[bi].item()) if margins is not None else 0.0,
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

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        if not self.distributed:
            self.model.process_task_count()
        else:
            self.model.module.process_task_count()
        self.task_id += 1
