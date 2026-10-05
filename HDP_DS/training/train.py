"""HDP-DS v3.1 训练模块: 训练循环、Checkpoint、Early Stopping"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainerConfig:
    """训练器配置"""
    max_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1e-6
    gradient_clip_val: float = 1.0
    checkpoint_dir: str = 'data/models'
    checkpoint_monitor: str = 'val_loss'
    checkpoint_mode: str = 'min'
    verbose: bool = True
    seed: int = 42


class Trainer:
    """HDP-DS 训练器: 管理训练、验证、checkpoint 保存"""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.current_epoch = 0
        self.best_metric = float('inf') if self.config.checkpoint_mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': []
        }

    def fit(
        self,
        model: nn.Module,
        train_loader: Any,       # DataLoader-like
        val_loader: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            model: PyTorch 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器

        Returns:
            training_history: {metric_name: [values]}
        """
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        device = next(model.parameters()).device

        for epoch in range(1, self.config.max_epochs + 1):
            self.current_epoch = epoch

            # ── 训练 ──
            model.train()
            train_loss = self._run_epoch(model, train_loader, optimizer, device, is_train=True)
            self.training_history['train_loss'].append(train_loss)

            # ── 验证 ──
            val_loss = float('inf')
            if val_loader is not None:
                val_loss = self._run_epoch(model, val_loader, None, device, is_train=False)
                self.training_history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

            # ── 日志 ──
            if self.config.verbose and epoch % 10 == 0:
                print(f"  [Train] Epoch {epoch:3d}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}")

            # ── Checkpoint ──
            monitor_value = val_loss if val_loader is not None else train_loss
            self._save_checkpoint(model, monitor_value)

            # ── Early Stopping ──
            if self._early_stop_check(monitor_value):
                if self.config.verbose:
                    print(f"  [Train] Early stop @ epoch {epoch}")
                break

            # ── Callbacks ──
            if callbacks:
                for cb in callbacks:
                    cb(self)

        if self.config.verbose:
            print(f"  [Train] 训练完成: {epoch} epochs, best at epoch {self.best_epoch}")

        return self.training_history

    def _run_epoch(
        self, model: nn.Module, loader: Any,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device, is_train: bool
    ) -> float:
        """运行一个 epoch"""
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            if is_train:
                optimizer.zero_grad()

            # 假设 batch 为 (static, dynamic, target)
            static_data, dynamic_data, target = batch

            static_data = static_data.to(device) if static_data is not None else None
            dynamic_data = dynamic_data.to(device)
            target = target.to(device)

            # 前向传播
            result = model(None, dynamic_data)  # TODO: 静态数据传入
            predictions = result['P']

            loss = nn.functional.mse_loss(predictions, target)

            if is_train:
                loss.backward()
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_val
                    )
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def _save_checkpoint(self, model: nn.Module, monitor_value: float):
        """保存 checkpoint"""
        is_better = False
        if self.config.checkpoint_mode == 'min':
            is_better = monitor_value < self.best_metric - self.config.early_stop_min_delta
        else:
            is_better = monitor_value > self.best_metric + self.config.early_stop_min_delta

        if is_better:
            self.best_metric = monitor_value
            self.best_epoch = self.current_epoch
            self.patience_counter = 0

            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / 'hdp_model_best.pt'
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,
                'best_metric': self.best_metric,
                'config': self.config
            }, checkpoint_path)
        else:
            self.patience_counter += 1

    def _early_stop_check(self, monitor_value: float) -> bool:
        """检查是否 early stop"""
        if self.patience_counter >= self.config.early_stop_patience:
            return True
        return False

    def save_training_history(self, path: str):
        """保存训练历史"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'history': self.training_history,
                'best_epoch': self.best_epoch,
                'best_metric': self.best_metric,
                'config': {
                    'max_epochs': self.config.max_epochs,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                }
            }, f, indent=2)
