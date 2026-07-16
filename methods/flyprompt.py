import gc
import json
import logging
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.router_eval import compute_router_metrics

from methods._trainer import _Trainer

logger = logging.getLogger()


class FlyPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(FlyPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.label_to_task: Dict[int, set] = {}

    def online_step(self, images, labels, idx):
        """单次在线训练入口：更新类别映射、执行 online_iter 次训练并收集路由统计。

        使用场景:
        - 每来一个在线 batch，就调用一次。
        - 适合单遍流式训练，不依赖 benchmark task 边界切换。

        小例子:
        - 若 ``online_iter=3``，同一个 batch 会被训练 3 次（轻量重复更新），
            然后再把该 batch 的特征统计写入 REAR 所需的累计量。
        """
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        self.collect(images.clone(), labels.clone())

        if hasattr(self, "_maybe_advance_internal_step"):
            batch_size_global = images.size(0) * self.world_size
            self._maybe_advance_internal_step(batch_size_global)

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def collect(self, images, labels):
        """把当前 batch 的样本写入路由统计，并维护 label -> expert 历史映射。

        使用场景:
        - 训练后调用，给 REAR 的统计矩阵补充新样本信息。
        - 需要在不反传的情况下更新模型内部统计。

        小例子:
        - 某类别第一次在当前 expert 出现，会在 ``label_to_task`` 中记录该 expert id；
            后续同类在其他 expert 出现时，这个集合会扩展为多 expert 关联。
        """
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            if label.item() not in self.label_to_task:
                self.label_to_task[label.item()] = set()
            self.label_to_task[label.item()].add(self.task_id)

        images = images.to(self.device)
        labels = labels.to(self.device)

        images = self.test_transform_tensor(images)

        with torch.no_grad():
            self.model.eval()
            self.model_without_ddp.collect(images, labels)

    def online_train(self, data):
        """对一个在线 batch 执行一次前向/反向更新，并更新 EMA 头。

        使用场景:
        - 在 ``online_step`` 内部被重复调用（由 ``online_iter`` 控制）。
        - 适合有 batch-class mask 的 GCL 训练。

        小例子:
        - 一个 batch 仅包含类别 {3, 7} 时，mask 会把其他类别 logit 压到 -inf，
            降低当前梯度对未出现类别的干扰。
        """
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

        # Update EMA heads for the expert corresponding to the current
        # internal step (model.task_count). This avoids using benchmark
        # task ids.
        if hasattr(self.model_without_ddp, "update_ema_fc"):
            self.model_without_ddp.update_ema_fc()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        """统一封装前向与损失计算，支持可选外部 mask。

        使用场景:
        - 训练阶段传入 batch 级 mask。
        - 其他情况下回退到全局 ``self.mask``。

        小例子:
        - 当 ``mask`` 为当前 batch 的 seen-class mask 时，模型只在该子空间内竞争。
        """
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        """在线评估：REAR 选 expert + TE2 多头聚合，输出平均与分类别精度。

        使用场景:
        - 周期性评估（如每 1000 个 batch）或训练结束后总评。
        - 不依赖 task oracle，只依赖路由器预测的 expert id。

        小例子:
        - 先用 ``forward_with_rp`` 得到每个样本的 expert id，
            再用 ``forward_with_ema`` 生成多头 logits 并聚合为最终预测。
        """
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        collect_head_analysis = (
            end
            and task_id is not None
            and task_id == self.n_tasks - 1
            and self.is_main_process()
        )
        head_total = None
        head_correct = None
        selected_count = None
        selected_correct = None
        pair_n = None
        pair_agree = None
        pair_both_correct = None
        pair_both_wrong = None
        pair_jsd_sum = None
        pair_confusion = None

        all_correct = []
        all_selected = []

        self.model_without_ddp.update()

        router_mode = getattr(self, 'router_mode', 'learned')
        log_router_trace = getattr(self, 'log_router_trace', False)
        analysis_router_quality = getattr(self, 'analysis_router_quality', False)

        self.model.eval()
        # If doing oracle-style modes, we need to accumulate logits per-route across dataset
        do_oracle_all = router_mode in ['sample_oracle', 'class_oracle', 'worst'] or analysis_router_quality
        results_summary = {}

        if do_oracle_all:
            # We'll accumulate per-route logits and labels across dataset
            n_experts = getattr(self.model_without_ddp, 'task_num', self.n_tasks)
            max_routes = getattr(self, 'oracle_max_routes', -1)
            if max_routes > 0:
                n_routes = min(n_experts, max_routes)
            else:
                n_routes = n_experts

            per_route_logits = [ [] for _ in range(n_routes) ]
            all_labels = []

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    x, y = data
                    for j in range(len(y)):
                        y[j] = self.exposed_classes.index(y[j].item())

                    x = x.to(self.device)
                    y = y.to(self.device)
                    batch_size = x.size(0)

                    # For each candidate route, force the expert and get ensembled logits
                    for r in range(n_routes):
                        expert_ids = torch.full((batch_size,), r, dtype=torch.long, device=self.device)
                        logit_ls = self.model_without_ddp.forward_with_ema(x, expert_ids=expert_ids)
                        if getattr(self, "eval_ema_only", False):
                            logit_ls = logit_ls[1:]
                        logit_ls = [logit + self.mask for logit in logit_ls]
                        logit = self._ensemble_logits(logit_ls)
                        per_route_logits[r].append(logit.detach().cpu())

                    all_labels.append(y.detach().cpu())

            # Concatenate per-route logits into full dataset tensors
            per_route_logits = [torch.cat(lst, dim=0) if len(lst)>0 else torch.empty((0,self.n_classes)) for lst in per_route_logits]
            all_labels = torch.cat(all_labels, dim=0)

            if self.is_main_process() and end:
                # Learned / Random / Single evaluated via separate pass below
                # Compute sample_oracle / class_oracle / worst using per_route_logits
                N = all_labels.size(0)
                labels_np = all_labels.numpy()

                # Sample-oracle
                stacked = torch.stack(per_route_logits, dim=0)  # [R, N, C]
                true_logits = stacked[:, :, :]
                # pick best route per sample by true-label logit
                true_label_indices = labels_np.astype(int)
                # build array [R, N] of true-label logits
                true_label_logits = np.stack([true_logits[r, np.arange(N), true_label_indices] .numpy() for r in range(stacked.size(0))], axis=0)
                best_r_per_sample = np.argmax(true_label_logits, axis=0)
                oracle_logits = np.zeros((N, self.n_classes), dtype=np.float32)
                for i_sample in range(N):
                    rsel = int(best_r_per_sample[i_sample])
                    oracle_logits[i_sample] = per_route_logits[rsel][i_sample].numpy()
                preds_oracle = np.argmax(oracle_logits, axis=1)
                A_sample_oracle = float((preds_oracle == labels_np).mean())

                # Worst router
                worst_r_per_sample = np.argmin(true_label_logits, axis=0)
                worst_logits = np.zeros((N, self.n_classes), dtype=np.float32)
                for i_sample in range(N):
                    rsel = int(worst_r_per_sample[i_sample])
                    worst_logits[i_sample] = per_route_logits[rsel][i_sample].numpy()
                preds_worst = np.argmax(worst_logits, axis=1)
                A_worst = float((preds_worst == labels_np).mean())

                # Class-oracle: choose best route per class by mean true-logit
                R = len(per_route_logits)
                C = self.n_classes
                class_best = np.zeros(C, dtype=int)
                for c in range(C):
                    idxs = np.where(labels_np == c)[0]
                    if idxs.size == 0:
                        class_best[c] = 0
                        continue
                    mean_true_logits = [per_route_logits[r][idxs, c].mean().item() for r in range(R)]
                    class_best[c] = int(np.argmax(mean_true_logits))

                class_oracle_logits = np.zeros((N, C), dtype=np.float32)
                for i_sample in range(N):
                    c = int(labels_np[i_sample])
                    rsel = int(class_best[c])
                    class_oracle_logits[i_sample] = per_route_logits[rsel][i_sample].numpy()
                preds_class_oracle = np.argmax(class_oracle_logits, axis=1)
                A_class_oracle = float((preds_class_oracle == labels_np).mean())

                results_summary['A_sample_oracle'] = A_sample_oracle
                results_summary['A_class_oracle'] = A_class_oracle
                results_summary['A_worst'] = A_worst
                # Save partial summary (learned/random/single will be added below)
            # end analysis_router_quality block

        # Non-oracle or learned/random/single evaluation path
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        label = []
        router_traces = []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                x = x.to(self.device)
                y = y.to(self.device)

                # use RP head to get expert_ids or random/single depending on mode
                logit_raw = self.model_without_ddp.forward_with_rp(x)
                if router_mode == 'learned':
                    expert_ids = torch.argmax(logit_raw, dim=-1)
                elif router_mode == 'random':
                    expert_ids = torch.randint(0, getattr(self.model_without_ddp, 'task_num', self.n_tasks), (x.size(0),), device=self.device)
                elif router_mode == 'single':
                    expert_ids = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                else:
                    # for oracle modes, we don't use this path
                    expert_ids = torch.argmax(logit_raw, dim=-1)

                logit_ls = self.model_without_ddp.forward_with_ema(x, expert_ids=expert_ids)

                if getattr(self, "eval_ema_only", False):
                    logit_ls = logit_ls[1:]

                logit_ls = [logit + self.mask for logit in logit_ls]

                if collect_head_analysis:
                    if head_total is None:
                        num_heads = len(logit_ls)
                        n_classes = logit_ls[0].size(-1)
                        head_total = torch.zeros(num_heads, dtype=torch.long)
                        head_correct = torch.zeros(num_heads, dtype=torch.long)
                        selected_count = torch.zeros(num_heads, dtype=torch.long)
                        selected_correct = torch.zeros(num_heads, dtype=torch.long)

                        pair_n = torch.zeros((num_heads, num_heads), dtype=torch.long)
                        pair_agree = torch.zeros((num_heads, num_heads), dtype=torch.long)
                        pair_both_correct = torch.zeros((num_heads, num_heads), dtype=torch.long)
                        pair_both_wrong = torch.zeros((num_heads, num_heads), dtype=torch.long)
                        pair_jsd_sum = torch.zeros((num_heads, num_heads), dtype=torch.float64)
                        pair_confusion = torch.zeros(
                            (num_heads, num_heads, n_classes, n_classes),
                            dtype=torch.long,
                        )

                    pred_per_head = []
                    prob_per_head = []
                    for h_idx, h_logit in enumerate(logit_ls):
                        h_pred = torch.argmax(h_logit, dim=-1)
                        pred_per_head.append(h_pred)
                        prob_per_head.append(torch.softmax(h_logit, dim=-1))
                        h_correct = (h_pred == y)
                        head_total[h_idx] += y.size(0)
                        head_correct[h_idx] += h_correct.sum().item()

                    pred_per_head = torch.stack(pred_per_head, dim=-1)
                    prob_stack = torch.stack(prob_per_head, dim=-1)
                    correct_per_head = pred_per_head.eq(y.unsqueeze(1))
                    selected_idx = self._select_head_indices_for_analysis(logit_ls)
                    batch_idx = torch.arange(selected_idx.size(0), device=selected_idx.device)
                    selected_pred = pred_per_head[batch_idx, selected_idx]
                    selected_is_correct = (selected_pred == y)

                    all_correct.append(correct_per_head.detach().cpu())
                    all_selected.append(selected_idx.detach().cpu())

                    selected_count += torch.bincount(
                        selected_idx.detach().cpu(),
                        minlength=selected_count.numel(),
                    )
                    selected_correct += torch.bincount(
                        selected_idx[selected_is_correct].detach().cpu(),
                        minlength=selected_correct.numel(),
                    )

                    for h_i in range(pred_per_head.size(-1)):
                        for h_j in range(h_i + 1, pred_per_head.size(-1)):
                            p_i = pred_per_head[:, h_i]
                            p_j = pred_per_head[:, h_j]

                            agree_ij = (p_i == p_j)
                            both_correct_ij = correct_per_head[:, h_i] & correct_per_head[:, h_j]
                            both_wrong_ij = (~correct_per_head[:, h_i]) & (~correct_per_head[:, h_j])

                            pair_n[h_i, h_j] += p_i.numel()
                            pair_n[h_j, h_i] += p_i.numel()
                            pair_agree[h_i, h_j] += agree_ij.sum().item()
                            pair_agree[h_j, h_i] += agree_ij.sum().item()
                            pair_both_correct[h_i, h_j] += both_correct_ij.sum().item()
                            pair_both_correct[h_j, h_i] += both_correct_ij.sum().item()
                            pair_both_wrong[h_i, h_j] += both_wrong_ij.sum().item()
                            pair_both_wrong[h_j, h_i] += both_wrong_ij.sum().item()

                            jsd_ij = self._batch_jsd(prob_stack[:, :, h_i], prob_stack[:, :, h_j])
                            pair_jsd_sum[h_i, h_j] += jsd_ij.sum().item()
                            pair_jsd_sum[h_j, h_i] += jsd_ij.sum().item()

                            flat_idx = (p_i * n_classes + p_j).detach().cpu()
                            conf_ij = torch.bincount(flat_idx, minlength=n_classes * n_classes)
                            conf_ij = conf_ij.view(n_classes, n_classes)
                            pair_confusion[h_i, h_j] += conf_ij
                            pair_confusion[h_j, h_i] += conf_ij.t()

                logit = self._ensemble_logits(logit_ls)

                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                # router trace logging
                if log_router_trace:
                    # logit_raw: [B, R]
                    topk_scores, topk_idx = torch.topk(logit_raw, k=min(2, logit_raw.size(1)), dim=-1)
                    topk_idx = topk_idx.detach().cpu().tolist()
                    topk_scores = topk_scores.detach().cpu().tolist()
                    margins = [s[0] - (s[1] if len(s) > 1 else -1e9) for s in topk_scores]
                    for bi in range(x.size(0)):
                        trace = {
                            'sample_id': None,
                            'label': int(y[bi].item()),
                            'pred': int(pred[bi].item()),
                            'correct': bool(pred[bi].item() == int(y[bi].item())),
                            'loss': float(loss.detach().cpu().item()),
                            'method': getattr(self, 'method', 'flyprompt'),
                            'dataset': getattr(self, 'dataset', 'cifar100'),
                            'seed': int(self.rnd_seed),
                            'eval_step': i,
                            'route_id': int(expert_ids[bi].item()),
                            'route_topk': topk_idx[bi],
                            'route_scores': topk_scores[bi],
                            'route_margin': float(margins[bi]),
                        }
                        router_traces.append(trace)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        # If oracle modes were precomputed and router_mode is oracle, derive the accuracy for this router_mode
        if do_oracle_all and router_mode in ['sample_oracle', 'class_oracle', 'worst'] and self.is_main_process() and end:
            # use previously computed values
            if router_mode == 'sample_oracle':
                avg_acc = results_summary.get('A_sample_oracle', avg_acc)
            elif router_mode == 'class_oracle':
                avg_acc = results_summary.get('A_class_oracle', avg_acc)
            elif router_mode == 'worst':
                avg_acc = results_summary.get('A_worst', avg_acc)

        # Save router trace if requested
        if log_router_trace and self.is_main_process():
            out_dir = os.path.join(self.log_dir, f"{self.dataset}", self.note)
            os.makedirs(out_dir, exist_ok=True)
            trace_path = os.path.join(out_dir, f"router_trace_seed_{self.rnd_seed}.pt")
            try:
                torch.save(router_traces, trace_path)
                logger.info("[FlyPrompt] Saved router trace to %s", trace_path)
            except Exception as e:
                logger.exception("[FlyPrompt] Failed to save router trace: %s", e)

        # If doing analysis, compute learned/random/single accuracies and save summary JSON
        if analysis_router_quality and self.is_main_process() and end:
            import json
            # evaluate learned/random/single accuracies
            def eval_mode_acc(mode):
                corr = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        x, y = data
                        for j in range(len(y)):
                            y[j] = self.exposed_classes.index(y[j].item())
                        x = x.to(self.device)
                        y = y.to(self.device)
                        logit_raw = self.model_without_ddp.forward_with_rp(x)
                        if mode == 'learned':
                            expert_ids = torch.argmax(logit_raw, dim=-1)
                        elif mode == 'random':
                            expert_ids = torch.randint(0, getattr(self.model_without_ddp, 'task_num', self.n_tasks), (x.size(0),), device=self.device)
                        elif mode == 'single':
                            expert_ids = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                        else:
                            expert_ids = torch.argmax(logit_raw, dim=-1)
                        logit_ls = self.model_without_ddp.forward_with_ema(x, expert_ids=expert_ids)
                        if getattr(self, "eval_ema_only", False):
                            logit_ls = logit_ls[1:]
                        logit_ls = [logit + self.mask for logit in logit_ls]
                        logit = self._ensemble_logits(logit_ls)
                        pred = torch.argmax(logit, dim=-1)
                        corr += (pred == y).sum().item()
                        total += y.size(0)
                return float(corr / max(1, total))

            A_learned = eval_mode_acc('learned')
            A_random = eval_mode_acc('random')
            A_single = eval_mode_acc('single')

            # integrate previous oracle results if present
            A_sample_oracle = results_summary.get('A_sample_oracle', None)
            A_class_oracle = results_summary.get('A_class_oracle', None)
            A_worst = results_summary.get('A_worst', None)

            # compute RQIs
            eps = 1e-8
            RQI_sample = (A_learned - A_random) / ( (A_sample_oracle - A_random) + eps) if A_sample_oracle is not None else None
            RQI_class = (A_learned - A_random) / ( (A_class_oracle - A_random) + eps) if A_class_oracle is not None else None

            # router metrics from traces (use learned routes)
            try:
                routes_used = [t['route_id'] for t in router_traces]
                labels_used = [t['label'] for t in router_traces]
                margins_used = [t['route_margin'] for t in router_traces]
                metrics = compute_router_metrics(labels_used, routes_used, margins_used, num_routes=getattr(self.model_without_ddp, 'task_num', self.n_tasks))
            except Exception as e:
                logger.exception("[FlyPrompt] Failed to compute router metrics: %s", e)
                metrics = {}

            summary = {
                "method": "flyprompt",
                "dataset": self.dataset,
                "seed": int(self.rnd_seed),
                "A_learned": A_learned,
                "A_random": A_random,
                "A_single": A_single,
                "A_class_oracle": A_class_oracle,
                "A_sample_oracle": A_sample_oracle,
                "A_worst": A_worst,
                "RQI_class": RQI_class,
                "RQI_sample": RQI_sample,
                "router_utility": A_learned - A_random,
                "oracle_gap_sample": (A_sample_oracle - A_learned) if A_sample_oracle is not None else None,
                "capacity_gap_sample": (A_sample_oracle - A_single) if A_sample_oracle is not None else None,
            }
            summary.update(metrics)

            out_dir = os.path.join(self.log_dir, f"{self.dataset}", self.note)
            os.makedirs(out_dir, exist_ok=True)
            summary_path = os.path.join(out_dir, f"router_quality_summary_seed_{self.rnd_seed}.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            logger.info("[FlyPrompt] Saved router quality summary to %s", summary_path)

        if collect_head_analysis and head_total is not None:
            detailed_analysis = self._build_head_agreement_analysis(
                head_correct=head_correct,
                head_total=head_total,
                pair_n=pair_n,
                pair_agree=pair_agree,
                pair_both_correct=pair_both_correct,
                pair_both_wrong=pair_both_wrong,
                pair_jsd_sum=pair_jsd_sum,
                pair_confusion=pair_confusion,
                all_correct=torch.cat(all_correct, dim=0),
                all_selected=torch.cat(all_selected, dim=0),
            )
            self._save_head_output_analysis(
                head_total=head_total,
                head_correct=head_correct,
                selected_count=selected_count,
                selected_correct=selected_correct,
                detailed_analysis=detailed_analysis,
            )

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def _select_head_indices_for_analysis(self, logit_ls):
        """Choose one head index per sample for analysis-only selection stats.

        - If ensemble method is min-entropy, select the head with smallest entropy.
        - Otherwise, select the head with highest max class probability.
        """
        prob_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]
        prob_stack = torch.stack(prob_ls, dim=-1)  # [B, C, H]

        if "min_entropy" in getattr(self, "ensemble_method", ""):
            ent = -torch.sum(prob_stack * torch.log(prob_stack + 1e-8), dim=1)  # [B, H]
            return torch.argmin(ent, dim=-1)

        max_prob = prob_stack.max(dim=1)[0]  # [B, H]
        return torch.argmax(max_prob, dim=-1)

    def _batch_jsd(self, p, q, eps=1e-8):
        """Compute sample-wise Jensen-Shannon divergence between two probability tensors.

        p, q: [B, C]
        return: [B]
        """
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p) - torch.log(m + eps)), dim=1)
        kl_qm = torch.sum(q * (torch.log(q) - torch.log(m + eps)), dim=1)
        return 0.5 * (kl_pm + kl_qm)

    def _cohen_kappa_from_confusion(self, conf):
        """Compute Cohen's kappa from a confusion matrix."""
        n = conf.sum().item()
        if n == 0:
            return 0.0
        p0 = conf.diag().sum().item() / n
        row = conf.sum(dim=1).float()
        col = conf.sum(dim=0).float()
        pe = (row * col).sum().item() / (n * n)
        if abs(1.0 - pe) < 1e-12:
            return 0.0
        return (p0 - pe) / (1.0 - pe)

    def _build_head_agreement_analysis(
        self,
        head_correct,
        head_total,
        pair_n,
        pair_agree,
        pair_both_correct,
        pair_both_wrong,
        pair_jsd_sum,
        pair_confusion,
        all_correct,
        all_selected,
    ):
        """Build detailed agreement/complementarity analysis across heads."""
        num_heads = head_total.numel()
        n_pair = pair_n.float().clamp_min(1.0)

        agreement_rate = (pair_agree.float() / n_pair).numpy()
        both_correct_rate = (pair_both_correct.float() / n_pair).numpy()
        double_fault_rate = (pair_both_wrong.float() / n_pair).numpy()
        mean_jsd = (pair_jsd_sum / n_pair.double()).numpy()

        head_acc = (head_correct.float() / head_total.float().clamp_min(1.0)).numpy()
        cohen_kappa = np.zeros((num_heads, num_heads), dtype=np.float64)
        for i in range(num_heads):
            for j in range(num_heads):
                if i == j:
                    cohen_kappa[i, j] = 1.0
                    agreement_rate[i, j] = 1.0
                    both_correct_rate[i, j] = head_acc[i]
                    double_fault_rate[i, j] = 1.0 - head_acc[i]
                    mean_jsd[i, j] = 0.0
                else:
                    cohen_kappa[i, j] = self._cohen_kappa_from_confusion(pair_confusion[i, j].float())

        per_head = []
        cond_correct = np.full((num_heads, num_heads), np.nan, dtype=np.float64)
        cond_wrong = np.full((num_heads, num_heads), np.nan, dtype=np.float64)

        for h in range(num_heads):
            mask_h = all_selected == h
            sel_correct_h = mask_h & all_correct[:, h]
            sel_wrong_h = mask_h & (~all_correct[:, h])

            correct_total = int(sel_correct_h.sum().item())
            wrong_total = int(sel_wrong_h.sum().item())

            other_ids = [i for i in range(num_heads) if i != h]
            if len(other_ids) > 0:
                other_correct_on_correct = all_correct[sel_correct_h][:, other_ids]
                other_correct_on_wrong = all_correct[sel_wrong_h][:, other_ids]

                hist_correct = (
                    torch.bincount(
                        other_correct_on_correct.sum(dim=1).long(), minlength=num_heads
                    )
                    if correct_total > 0
                    else torch.zeros(num_heads, dtype=torch.long)
                )
                hist_wrong = (
                    torch.bincount(
                        other_correct_on_wrong.sum(dim=1).long(), minlength=num_heads
                    )
                    if wrong_total > 0
                    else torch.zeros(num_heads, dtype=torch.long)
                )

                for j in other_ids:
                    if correct_total > 0:
                        cond_correct[h, j] = (
                            (all_correct[:, j] & sel_correct_h).sum().item() / correct_total
                        )
                    if wrong_total > 0:
                        cond_wrong[h, j] = (
                            (all_correct[:, j] & sel_wrong_h).sum().item() / wrong_total
                        )
            else:
                hist_correct = torch.zeros(num_heads, dtype=torch.long)
                hist_wrong = torch.zeros(num_heads, dtype=torch.long)

            all_other_correct_idx = max(0, num_heads - 1)
            per_head.append(
                {
                    "head_index": h,
                    "selected_correct_total": correct_total,
                    "selected_wrong_total": wrong_total,
                    "selected_correct_exactly_k_other_correct": [
                        int(v) for v in hist_correct.tolist()
                    ],
                    "selected_wrong_exactly_k_other_correct": [
                        int(v) for v in hist_wrong.tolist()
                    ],
                    "selected_correct_all_other_correct_rate": (
                        float(hist_correct[all_other_correct_idx].item() / max(1, correct_total))
                    ),
                    "selected_wrong_none_other_correct_rate": (
                        float(hist_wrong[0].item() / max(1, wrong_total))
                    ),
                }
            )

        return {
            "pairwise": {
                "agreement_rate": agreement_rate.tolist(),
                "both_correct_rate": both_correct_rate.tolist(),
                "double_fault_rate": double_fault_rate.tolist(),
                "mean_jsd": mean_jsd.tolist(),
                "cohen_kappa": cohen_kappa.tolist(),
            },
            "conditional": {
                "other_correct_given_selected_correct": cond_correct.tolist(),
                "other_correct_given_selected_wrong": cond_wrong.tolist(),
                "per_head": per_head,
            },
        }

    def _save_head_output_analysis(
        self,
        head_total,
        head_correct,
        selected_count,
        selected_correct,
        detailed_analysis=None,
    ):
        """Save per-head standalone accuracy and selection-conditioned accuracy.

        Artifacts are saved under self.log_dir as JSON and PNG figures.
        """
        os.makedirs(self.log_dir, exist_ok=True)

        head_total_f = head_total.float().clamp_min(1.0)
        selected_count_f = selected_count.float().clamp_min(1.0)

        standalone_acc = (head_correct.float() / head_total_f).cpu().numpy()
        selected_acc = (selected_correct.float() / selected_count_f).cpu().numpy()

        if getattr(self, "eval_ema_only", False):
            head_names = [f"ema_{i+1}" for i in range(len(standalone_acc))]
        else:
            head_names = ["online"] + [f"ema_{i+1}" for i in range(max(0, len(standalone_acc) - 1))]
        analysis = {
            "note": self.note,
            "seed": int(self.rnd_seed),
            "ensemble_method": getattr(self, "ensemble_method", "softmax_max_prob"),
            "head_names": head_names,
            "standalone": {
                "total": head_total.tolist(),
                "correct": head_correct.tolist(),
                "acc": [float(x) for x in standalone_acc],
            },
            "selected": {
                "count": selected_count.tolist(),
                "correct": selected_correct.tolist(),
                "acc_when_selected": [float(x) for x in selected_acc],
                "overall_selected_acc": float(selected_correct.sum().item() / max(1, selected_count.sum().item())),
            },
        }
        if detailed_analysis is not None:
            analysis["detailed"] = detailed_analysis

        json_path = os.path.join(self.log_dir, f"head_output_analysis_seed_{self.rnd_seed}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        logger.info("[FlyPrompt] Saved head output analysis to %s", json_path)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x = np.arange(len(head_names))

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(x, standalone_acc, color="#2e7d32")
            ax.set_xticks(x)
            ax.set_xticklabels(head_names)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Accuracy")
            ax.set_title("Per-head standalone accuracy (no selection)")
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"head_standalone_acc_seed_{self.rnd_seed}.png"))
            plt.close(fig)

            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.bar(x - 0.18, selected_count.cpu().numpy(), width=0.36, color="#1565c0", label="selected count")
            ax2 = ax1.twinx()
            ax2.bar(x + 0.18, selected_acc, width=0.36, color="#ef6c00", label="acc when selected")

            ax1.set_xticks(x)
            ax1.set_xticklabels(head_names)
            ax1.set_ylabel("Selection count")
            ax2.set_ylabel("Accuracy when selected")
            ax2.set_ylim(0.0, 1.0)
            ax1.set_title("Head selection count and accuracy when selected")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"head_selection_stats_seed_{self.rnd_seed}.png"))
            plt.close(fig)

            if detailed_analysis is not None:
                pair = detailed_analysis["pairwise"]
                pair_titles = [
                    ("agreement_rate", "Pairwise prediction agreement"),
                    ("cohen_kappa", "Pairwise Cohen kappa"),
                    ("mean_jsd", "Pairwise mean JSD (lower is closer)"),
                ]

                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
                for ax, (key, title) in zip(axes, pair_titles):
                    mat = np.array(pair[key], dtype=np.float64)
                    im = ax.imshow(mat, cmap="viridis")
                    ax.set_xticks(x)
                    ax.set_xticklabels(head_names, rotation=30, ha="right")
                    ax.set_yticks(x)
                    ax.set_yticklabels(head_names)
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(os.path.join(self.log_dir, f"head_pairwise_metrics_seed_{self.rnd_seed}.png"))
                plt.close(fig)

                cond = detailed_analysis["conditional"]
                cond_titles = [
                    (
                        "other_correct_given_selected_correct",
                        "P(other head correct | selected head correct)",
                    ),
                    (
                        "other_correct_given_selected_wrong",
                        "P(other head correct | selected head wrong)",
                    ),
                ]
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                for ax, (key, title) in zip(axes, cond_titles):
                    mat = np.array(cond[key], dtype=np.float64)
                    im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=1.0)
                    ax.set_xticks(x)
                    ax.set_xticklabels(head_names, rotation=30, ha="right")
                    ax.set_yticks(x)
                    ax.set_yticklabels(head_names)
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(os.path.join(self.log_dir, f"head_conditional_metrics_seed_{self.rnd_seed}.png"))
                plt.close(fig)
        except Exception as e:
            logger.exception("[FlyPrompt] Failed to plot head output analysis: %s", e)

    def _ensemble_logits(self, logit_ls):
        """聚合 online/EMA 多头输出。

        使用场景:
        - FlyPrompt 推理阶段的 TE2 头融合。
        - 可在 ``mean`` / ``max_prob`` / ``min_entropy`` 间切换。

        小例子:
        - 默认 ``softmax_max_prob``：先对每个头做 softmax，再按类别取逐元素最大值。
        """
        if not hasattr(self, 'ensemble_method'):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # Shape: [batch_size, n_classes, n_experts]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)  # [batch_size, n_experts]
            min_entropy_indices = torch.argmin(entropies, dim=-1)  # [batch_size]
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        """Keep benchmark task id for logging; model step advances by samples."""
        self.task_id += 1

    def analyze_expert_features(self):
        """Extract per-expert CLS features on the full test set, compute
        similarity and CKA matrices, and save them (plus heatmaps) under
        self.log_dir for the current seed.

        This is called once at the end of training from _Trainer.main_worker
        on the main process only.

        使用场景:
        - 训练完成后做可解释性分析，观察 expert 是否分工。
        - 对比“原始特征”与“残差特征（去公共 backbone 成分）”的相似度。

        小例子:
        - 如果 residual CKA 的非对角项较低，通常说明不同 expert 在学习互补子空间。
        """
        if not hasattr(self, "model_without_ddp"):
            logger.warning("[FlyPrompt] model_without_ddp not found, skip expert analysis.")
            return

        model = self.model_without_ddp
        model.eval()

        if not hasattr(self, "test_dataset"):
            logger.warning("[FlyPrompt] test_dataset not found, skip expert analysis.")
            return

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

        # Infer feature dimension using a small probe batch
        with torch.no_grad():
            sample_x, _ = next(iter(test_loader))
            sample_x = sample_x.to(device)
            # Use expert 0 just to probe the dimension
            probe_ids = torch.zeros(sample_x.size(0), dtype=torch.long, device=device)
            sample_feat = model.experts(model.backbone, sample_x, probe_ids)
            feat_dim = sample_feat.size(-1)

        num_samples = len(self.test_dataset)
        features = torch.zeros(n_experts, num_samples, feat_dim, dtype=torch.float32)
        common_features = torch.zeros(num_samples, feat_dim, dtype=torch.float32)

        logger.info(
            f"[FlyPrompt] Extracting CLS features for {n_experts} experts over "
            f"{num_samples} test samples (dim={feat_dim}) ..."
        )

        offset = 0
        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                batch_size = x.size(0)
                x = x.to(device)

                idx_slice = slice(offset, offset + batch_size)

                # Common representation: frozen backbone CLS without any expert prompts.
                common_batch = model.backbone.forward_features(x)[:, 0]
                common_features[idx_slice, :] = common_batch.detach().cpu()

                for t in range(n_experts):
                    expert_ids = torch.full(
                        (batch_size,), t, dtype=torch.long, device=device
                    )
                    feat_t = model.experts(model.backbone, x, expert_ids)
                    features[t, idx_slice, :] = feat_t.detach().cpu()

                offset += batch_size

        # Save raw features for potential further analysis
        os.makedirs(self.log_dir, exist_ok=True)
        feat_path = os.path.join(self.log_dir, f"features_seed_{self.rnd_seed}.pt")
        torch.save(features, feat_path)
        logger.info(f"[FlyPrompt] Saved expert features to {feat_path}")

        # Also save the common (backbone-only) features used for residual analysis
        common_feat_path = os.path.join(
            self.log_dir, f"common_features_seed_{self.rnd_seed}.pt"
        )
        torch.save(common_features, common_feat_path)
        logger.info(
            f"[FlyPrompt] Saved common (backbone-only) features to {common_feat_path}"
        )

        # ---------- Cosine similarity between per-task mean features ----------
        mean_feats = features.mean(dim=1)  # [T, D]
        mean_norm = mean_feats / (mean_feats.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = mean_norm @ mean_norm.t()  # [T, T]

        sim_path = os.path.join(self.log_dir, f"similarity_seed_{self.rnd_seed}.npy")
        np.save(sim_path, sim_matrix.numpy())
        logger.info(f"[FlyPrompt] Saved expert similarity matrix to {sim_path}")

        # Plot heatmap for similarity matrix if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt expert similarity (cosine of mean CLS)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            sim_fig_path = os.path.join(
                self.log_dir, f"similarity_seed_{self.rnd_seed}.png"
            )
            fig.savefig(sim_fig_path)
            plt.close(fig)
            logger.info(f"[FlyPrompt] Saved similarity heatmap to {sim_fig_path}")
        except Exception as e:
            logger.exception(
                "[FlyPrompt] Failed to plot similarity heatmap: %s", e
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
        # For CKA we work on all CUB200 test samples (small enough).
        for i in range(n_experts):
            x = features[i]  # [N, D]
            for j in range(n_experts):
                y = features[j]
                cka_matrix[i, j] = _cka(x, y)

        cka_path = os.path.join(self.log_dir, f"cka_seed_{self.rnd_seed}.npy")
        np.save(cka_path, cka_matrix.numpy())
        logger.info(f"[FlyPrompt] Saved expert CKA matrix to {cka_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt expert CKA")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            cka_fig_path = os.path.join(self.log_dir, f"cka_seed_{self.rnd_seed}.png")
            fig.savefig(cka_fig_path)
            plt.close(fig)
            logger.info(f"[FlyPrompt] Saved CKA heatmap to {cka_fig_path}")
        except Exception as e:
            logger.exception("[FlyPrompt] Failed to plot CKA heatmap: %s", e)

        # ---------- Residual expert analysis: (expert - mean over experts) ----------
        # Ref shape: [1, N, D], residual shape: [T, N, D]
        ref = features.mean(dim=0, keepdim=True)
        residual = features - ref

        residual_feat_path = os.path.join(
            self.log_dir, f"residual_features_seed_{self.rnd_seed}.pt"
        )
        torch.save(residual, residual_feat_path)
        logger.info(
            f"[FlyPrompt] Saved residual expert features to {residual_feat_path}"
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
            f"[FlyPrompt] Saved residual expert similarity matrix to {residual_sim_path}"
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
            ax.set_title("FlyPrompt residual expert similarity (cosine of mean CLS)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            residual_sim_fig_path = os.path.join(
                self.log_dir, f"residual_similarity_seed_{self.rnd_seed}.png"
            )
            fig.savefig(residual_sim_fig_path)
            plt.close(fig)
            logger.info(
                f"[FlyPrompt] Saved residual similarity heatmap to {residual_sim_fig_path}"
            )
        except Exception as e:
            logger.exception(
                "[FlyPrompt] Failed to plot residual similarity heatmap: %s", e
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
            f"[FlyPrompt] Saved residual expert CKA matrix to {residual_cka_path}"
        )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(residual_cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt residual expert CKA")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            residual_cka_fig_path = os.path.join(
                self.log_dir, f"residual_cka_seed_{self.rnd_seed}.png"
            )
            fig.savefig(residual_cka_fig_path)
            plt.close(fig)
            logger.info(
                f"[FlyPrompt] Saved residual CKA heatmap to {residual_cka_fig_path}"
            )
        except Exception as e:
            logger.exception("[FlyPrompt] Failed to plot residual CKA heatmap: %s", e)
