"""HDP-DS v3.1 损失函数模块: MultiTaskLoss, RMSELoss, QuantileLoss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiTaskLoss(nn.Module):
    """多任务加权损失 (Uncertainty Weighting): L_total = Σ(L_i * exp(-s_i) + s_i / 2)"""

    def __init__(self, num_tasks: int = 5, init_log_var: float = 0.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: {task_name: [batch, 1]} 或 [batch, num_tasks]
            targets: [batch, num_tasks]
            task_names: 任务名称列表

        Returns:
            total_loss: 加权总损失
            per_task_losses: 每个任务的损失
        """
        if isinstance(predictations, dict):
            preds = torch.cat(
                [predictions[name].squeeze(-1) for name in predictions],
                dim=-1
            )
        else:
            preds = predictions

        if task_names is None:
            task_names = [f"task_{i}" for i in range(self.num_tasks)]

        per_task_losses = {}
        weighted_losses = []

        for i in range(self.num_tasks):
            task_loss = F.mse_loss(preds[:, i], targets[:, i], reduction='mean')
            per_task_losses[task_names[i]] = task_loss

            # Uncertainty Weighting
            precision = torch.exp(-self.log_vars[i])
            weighted_losses.append(
                precision * task_loss + self.log_vars[i] / 2.0
            )

        total_loss = sum(weighted_losses)
        return total_loss, per_task_losses


class RMSELoss(nn.Module):
    """RMSE 损失"""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


class QuantileLoss(nn.Module):
    """
    分位数损失 (用于 TFT 概率预测)

    Q(y, y_hat, τ) = max(τ * (y - y_hat), (τ - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [batch, n_quantiles]
            target: [batch, 1]
        """
        n_quantiles = len(self.quantiles)
        target_expanded = target.expand_as(pred)

        losses = []
        for i, tau in enumerate(self.quantiles):
            error = target_expanded[:, i] - pred[:, i]
            loss = torch.max(
                tau * error,
                (tau - 1) * error
            )
            losses.append(loss.mean())

        return torch.stack(losses).mean()
