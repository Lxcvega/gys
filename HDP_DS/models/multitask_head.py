"""HDP-DS v3.1 Multi-task Prediction Head — 多任务预测头"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MultiTaskConfig:
    """多任务头配置"""
    embedding_dim: int = 64
    num_tasks: int = 5
    hidden_dims: List[int] = None
    task_hidden_dim: int = 16
    use_uncertainty_weighting: bool = True
    output_lower: float = 0.0
    output_upper: float = 1.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


# ─── 任务特定分支 ───────────────────────────────────────────────

class TaskSpecificHead(nn.Module):
    """单个任务的预测头"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskPredictionHead(nn.Module):
    """多任务预测头: 共享主干 + 任务特定分支 + Uncertainty Weighting"""

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        self.num_tasks = config.num_tasks

        layers, prev_dim = [], config.embedding_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        self.shared_backbone = nn.Sequential(*layers)

        self.task_heads = nn.ModuleList([TaskSpecificHead(prev_dim, config.task_hidden_dim) for _ in range(config.num_tasks)])

        if config.use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(config.num_tasks))

        self.task_names = ['Delivery Reliability', 'Quality Stability', 'Environmental Risk', 'Cost Stability', 'Service Capability']

    def forward(self, Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Z: [batch, embedding_dim] → {task_name: [batch, 1]}"""
        shared = self.shared_backbone(Z)
        return {self.task_names[i]: torch.sigmoid(self.task_heads[i](shared)) for i in range(self.num_tasks)}

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务加权损失

        Args:
            predictions: {task_name: [batch, 1]}
            targets: [batch, 5] (按任务顺序)

        Returns:
            total_loss: 加权总损失
            losses: 每个任务的损失字典
        """
        losses = {}
        task_losses = []

        for i, task_name in enumerate(self.task_names):
            pred = predictions[task_name].squeeze(-1)  # [batch]
            target = targets[:, i]                       # [batch]

            # MSE 损失
            loss = F.mse_loss(pred, target, reduction='mean')
            losses[task_name] = loss
            task_losses.append(loss)

        # 不确定性加权
        if self.config.use_uncertainty_weighting:
            # L_total = Σ_i (L_i * exp(-s_i) + s_i / 2)
            # 其中 s_i = log(σ_i^2)
            weighted_losses = []
            for i, loss in enumerate(task_losses):
                precision = torch.exp(-self.log_vars[i])
                weighted_losses.append(
                    precision * loss + self.log_vars[i] / 2.0
                )
            total_loss = sum(weighted_losses)
        else:
            # 简单平均
            total_loss = torch.mean(torch.stack(task_losses))

        return total_loss, losses

    def get_task_weights(self) -> Dict[str, float]:
        """获取任务权重 (不确定性加权)"""
        if not self.config.use_uncertainty_weighting:
            return {name: 1.0 / self.num_tasks for name in self.task_names}

        weights = {}
        log_vars = F.softplus(self.log_vars.detach())  # 确保正数
        total = log_vars.sum().item()
        for i, name in enumerate(self.task_names):
            weights[name] = (log_vars[i].item() / total) if total > 0 else 1.0 / self.num_tasks
        return weights


# ─── 单任务预测头 (用于消融实验 — 当 use_multitask=False) ─────

class SingleTaskHead(nn.Module):
    """单任务预测头 (替代多任务，用于消融对比)"""

    def __init__(self, embedding_dim: int, num_outputs: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
