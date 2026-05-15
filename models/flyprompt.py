import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class Prompt(nn.Module):
    """Prompt expert 容器：管理多 expert、多层 prompt 参数。

    何时调用:
    - 由 `models.FlyPrompt.forward/forward_with_ema` 间接调用。
    - 训练和评估只要走 expert prompt 路径都会用到。
    """

    def __init__(self,
                 num_experts: int,
                 len_prompt: int = 20,
                 embed_dim: int = 768,
                 pos_prompt: Iterable[int] = (0, 1, 2, 3, 4)):
        super().__init__()
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim

        self.register_buffer('pos_prompt', torch.tensor(list(pos_prompt), dtype=torch.int64))
        self.num_layers = int(self.pos_prompt.numel())

        self.prompts = nn.Parameter(
            torch.empty(self.num_layers, num_experts, len_prompt, embed_dim)
        )
        nn.init.uniform_(self.prompts)

    def _build_batched_prompts(self, backbone: nn.Module, expert_ids: torch.Tensor) -> torch.Tensor:
        """按 batch 的 expert_ids 组装对应 prompt 张量。

        何时调用:
        - 仅在 `Prompt.forward` 内部调用。

        小例子:
        - 若 batch 中第 0 个样本 expert_id=2，则该样本会选第 2 号 expert 的各层 prompt。
        """
        B = expert_ids.size(0)
        prompts = []
        for l_idx in range(self.num_layers):
            p_l = self.prompts[l_idx][expert_ids.long()]  # [B, len_prompt, D]
            prompts.append(p_l)
        prompts = torch.stack(prompts, dim=1)  # [B, num_layers, len_prompt, D]

        D = prompts.size(-1)
        pos_bias = backbone.pos_embed[:, :1, :].unsqueeze(1).expand(B, self.num_layers, self.len_prompt, D)
        prompts = prompts + pos_bias
        return prompts

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """把 prompt 注入 ViT 指定层并输出 CLS 特征。

        何时调用:
        - 在 `models.FlyPrompt.forward` 和 `models.FlyPrompt.forward_with_ema` 中调用。
        - 对应方法层调用点通常来自 `methods/flyprompt.py` 的训练与评估流程。

        小例子:
        - 当 `pos_prompt=(0,1,2,3,4)` 时，只在前 5 个 transformer block 注入 prompt token。
        """
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(token_appended + backbone.pos_embed)
        orig_N = x.size(1)

        prompts = self._build_batched_prompts(backbone, expert_ids)  # [B, num_layers, len_prompt, D]

        for n, block in enumerate(backbone.blocks):
            pos_n = (self.pos_prompt.eq(n)).nonzero(as_tuple=False).squeeze()
            if pos_n.numel() != 0:
                x = torch.cat((x, prompts[:, pos_n]), dim=1)
            x = block(x)
            x = x[:, :orig_N, :]

        x = backbone.norm(x)
        return x[:, 0]

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        """初始化新 expert 的 prompt 参数（均值 warm start）。

        何时调用:
        - 在 `models.FlyPrompt.process_task_count` 里调用，通常发生在内部 step 切换时。
        """
        if expert_id == 0 or expert_id >= self.num_experts:
            return
        prev_experts = self.prompts[:, :expert_id].clone()  # [num_layers, expert_id, L, D]
        prev_experts_mean = prev_experts.mean(dim=1)        # [num_layers, L, D]
        self.prompts.data[:, expert_id] = prev_experts_mean


class RPFC(nn.Module):
    """REAR 路由头：随机投影 + 闭式岭回归。

    何时调用:
    - `collect()` 在训练期间累计统计量 G/Q。
    - `update()` 在评估前（或 step 切换时）把统计量解成路由器权重。
    - `forward()` 在评估时根据特征预测 expert 打分。
    """

    def __init__(self,
                 M            : int,
                 ridge        : float = 1e4,
                 embed_dim    : int = 768,
                 num_classes  : int = 100,
                 **kwargs):

        super().__init__()
        
        self.ridge = ridge
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if M == 0:
            self.M = embed_dim
            self.use_rp = False
            self.register_buffer('W_rand', torch.empty(0))
            self.register_buffer('Q', torch.zeros(embed_dim, num_classes))
            self.register_buffer('G', torch.zeros(embed_dim, embed_dim))
        else:
            self.M = M
            self.use_rp = True
            self.register_buffer('W_rand', torch.randn(embed_dim, M))
            self.register_buffer('Q', torch.zeros(M, num_classes))
            self.register_buffer('G', torch.zeros(M, M))

        self.fc = nn.Linear(self.M, num_classes, bias=False)

        for param in self.parameters():
            param.requires_grad = False

    def target2onehot(self, targets):
        """把 expert 标签索引转成 one-hot。"""
        device = targets.device
        onehot = torch.zeros(targets.size(0), self.num_classes, device=device)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        return onehot

    def collect(self, features, labels):
        """累计闭式解所需统计量 Q/G。

        何时调用:
        - 来自 `models.FlyPrompt.collect`，进一步由 `methods/flyprompt.py::collect` 触发。
        """
        features = features.detach()
        labels = labels.detach()

        if self.use_rp:
            features_h = F.relu(features @ self.W_rand)
        else:
            features_h = features
        Y = self.target2onehot(labels)
        self.Q = self.Q + features_h.T @ Y
        self.G = self.G + features_h.T @ features_h

    def update(self):
        """根据当前 G/Q 求解岭回归权重并写入 fc。"""
        device = self.fc.weight.device
        Wo = torch.linalg.solve(self.G + self.ridge * torch.eye(self.M, device=device), self.Q).T
        self.fc.weight.data = Wo.to(device)

    def forward(self, x):
        """输出每个 expert 的路由分数。"""
        if self.use_rp:
            x = F.relu(x @ self.W_rand)
        x = self.fc(x)
        return x


class FlyPrompt(nn.Module):
    """FlyPrompt 模型主体：backbone + prompt experts + RP 路由 + EMA 头。

    何时调用:
    - 由 `utils/train_utils.select_model` 构建，随后挂到 trainer 的 `self.model`。
    - 训练与评估均通过本类方法进行前向、路由和统计更新。
    """

    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 backbone_name  : str   = None,
                 len_prompt     : int   = 20,
                 pos_prompt     : Iterable[int] = (0, 1, 2, 3, 4),
                 rp_dim         : int   = 10000,
                 rp_ridge       : float = 1e4,
                 ema_ratio      : Iterable[float] = (0.9, 0.99),
                 **kwargs):

        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes
        self.len_prompt = len_prompt
        self.pos_prompt = pos_prompt
        self.rp_dim = rp_dim
        self.rp_ridge = rp_ridge
        self.ema_ratio = ema_ratio
        self.num_ema = len(ema_ratio)

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Expert prompts
        self.experts = Prompt(
            num_experts = self.task_num,
            len_prompt = self.len_prompt,
            embed_dim = self.embed_dim,
            pos_prompt = self.pos_prompt,
        )

        # Expert FCs
        self.experts_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.embed_dim, self.num_classes, bias=True) for _ in range(self.num_ema)
            ]) for _ in range(self.task_num)
        ])
        for expert_fc in self.experts_fc:
            for fc in expert_fc:
                for param in fc.parameters():
                    param.requires_grad = False
        self.init_fc(expert_id = 0)

        # Per-expert frozen snapshots of the online classifier head.
        # These are updated at task boundaries and used at evaluation time.
        self.experts_online_fc = nn.ModuleList([
            nn.Linear(self.embed_dim, self.num_classes, bias=True) for _ in range(self.task_num)
        ])
        for fc in self.experts_online_fc:
            for param in fc.parameters():
                param.requires_grad = False
        self.snapshot_online_fc(expert_id=0)

        # Random projection head
        self.rp_head = RPFC(
            M = self.rp_dim,
            ridge = self.rp_ridge,
            embed_dim = self.embed_dim,
            num_classes = self.task_num,
        )

    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """标准训练前向：当前 expert 的 prompt 特征 + online 分类头。

        何时调用:
        - 由 `methods/flyprompt.py::model_forward` 中的 `self.model(x)` 触发。
        """
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x
    
    def forward_with_rp(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """路由前向：输出 expert 选择分数。

        何时调用:
        - 由 `methods/flyprompt.py::online_evaluate` 调用，用于选 expert_ids。
        """
        x = self.backbone.forward_features(inputs)
        x = x[:, 0]
        x = self.rp_head(x)
        return x
    
    def forward_with_ema(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """多头前向：返回 online head + 多个 EMA head 的 logits 列表。

        何时调用:
        - 由 `methods/flyprompt.py::online_evaluate` 调用，再交给 `_ensemble_logits` 聚合。

        小例子:
        - 返回长度为 `1 + num_ema` 的列表，第 0 个是 online 头，后续是 EMA 头。
        """
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        outputs_ls = []

        # online head: use task-specific snapshot for finished experts,
        # and current trainable online head for the in-progress expert.
        online_outputs = []
        for x_i, e_i in zip(x, expert_ids):
            e_id = e_i.item()
            if e_id == self.task_count:
                online_outputs.append(self.backbone.fc(x_i))
            else:
                online_outputs.append(self.experts_online_fc[e_id](x_i))
        outputs_ls.append(torch.stack(online_outputs, dim=0))
        
        # ema head
        for i in range(self.num_ema):
            outputs = []
            for x_i, e_i in zip(x, expert_ids):
                outputs.append(self.experts_fc[e_i.item()][i](x_i))
            outputs = torch.stack(outputs, dim=0)
            outputs_ls.append(outputs)

        return outputs_ls
    
    def collect(self, inputs: torch.Tensor, labels: torch.Tensor):
        """训练期收集路由统计：把样本特征记到账本 G/Q。

        何时调用:
        - 由 `methods/flyprompt.py::collect` 在每个在线 batch 后调用。
        """
        features = self.backbone.forward_features(inputs)
        features = features[:, 0]
        labels = torch.full((labels.size(0),), self.task_count, device=labels.device, dtype=torch.long)
        self.rp_head.collect(features, labels)

    def update(self):
        """刷新路由器权重。

        何时调用:
        - 评估前由 `methods/flyprompt.py::online_evaluate` 调用。
        - 内部 step 前进时也会在 `process_task_count` 里调用。
        """
        self.rp_head.update()

    @torch.no_grad()
    def init_fc(self, expert_id: int = None):
        """把当前 online 分类头参数拷贝给指定 expert 的 EMA 头初始化。"""
        if expert_id is None:
            expert_id = self.task_count
        if expert_id >= self.task_num:
            return
        w, b = self.backbone.fc.weight.data, self.backbone.fc.bias.data
        for i in range(self.num_ema):
            self.experts_fc[expert_id][i].weight.data.copy_(w)
            self.experts_fc[expert_id][i].bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self, expert_id: int = None):
        """更新指定 expert 的 EMA 分类头。

        何时调用:
        - 由 `methods/flyprompt.py::online_train` 在每个优化 step 后调用。
        """
        if expert_id is None:
            expert_id = self.task_count
        for i in range(self.num_ema):
            ema_ratio = self.ema_ratio[i]
            online_w = self.backbone.fc.weight.data
            online_b = self.backbone.fc.bias.data
            ema_w = self.experts_fc[expert_id][i].weight.data
            ema_b = self.experts_fc[expert_id][i].bias.data
            ema_w.mul_(ema_ratio).add_(online_w, alpha=1.0 - ema_ratio)
            ema_b.mul_(ema_ratio).add_(online_b, alpha=1.0 - ema_ratio)

    @torch.no_grad()
    def snapshot_online_fc(self, expert_id: int = None):
        """保存当前 online 分类头到指定 expert 的快照头。"""
        if expert_id is None:
            expert_id = self.task_count
        if expert_id >= self.task_num:
            return
        self.experts_online_fc[expert_id].weight.data.copy_(self.backbone.fc.weight.data)
        self.experts_online_fc[expert_id].bias.data.copy_(self.backbone.fc.bias.data)

    def loss_fn(self, output, target):
        """训练损失接口，供 trainer 侧统一调用。"""
        return F.cross_entropy(output, target)

    def process_task_count(self):
        """内部 step 前进时的状态切换：切换 expert、刷新路由、初始化新 expert 参数。

        何时调用:
        - 由 `_Trainer._maybe_advance_internal_step` 触发（按样本数推进）。
        """
        self.task_count += 1
        self.rp_head.update()
        self.experts.init_new_expert(self.task_count)
        self.init_fc(self.task_count)
        self.snapshot_online_fc(self.task_count)