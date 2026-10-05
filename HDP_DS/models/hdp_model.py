"""HDP-DS v3.1 HDP Model — 完整模型装配"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from .static_encoder import XGBoostStaticEncoder, StaticEncoderConfig
from .tft_encoder import TFTDynamicEncoder, TFTConfig
from .fusion import CrossAttentionFusion, FusionConfig, SimpleFusion
from .multitask_head import MultiTaskPredictionHead, MultiTaskConfig, SingleTaskHead


@dataclass
class HDPModelConfig:
    """HDP 完整模型配置"""
    static_encoder_config: StaticEncoderConfig = None
    tft_config: TFTConfig = None
    fusion_config: FusionConfig = None
    multitask_config: MultiTaskConfig = None
    use_static: bool = True
    use_dynamic: bool = True
    use_attention: bool = True
    use_multitask: bool = True
    static_embedding_dim: int = 32
    dynamic_embedding_dim: int = 32
    fusion_embedding_dim: int = 64
    learning_rate: float = 1e-3
    device: str = 'auto'

    def __post_init__(self):
        if self.static_encoder_config is None:
            self.static_encoder_config = StaticEncoderConfig(embedding_dim=self.static_embedding_dim)
        if self.tft_config is None:
            self.tft_config = TFTConfig(embedding_dim=self.dynamic_embedding_dim)
        if self.fusion_config is None:
            self.fusion_config = FusionConfig(
                static_dim=self.static_embedding_dim, dynamic_dim=self.dynamic_embedding_dim,
                embedding_dim=self.fusion_embedding_dim
            )
        if self.multitask_config is None:
            self.multitask_config = MultiTaskConfig(embedding_dim=self.fusion_embedding_dim)


class HDPModel(nn.Module):
    """HDP-DS v3.1 完整模型: Static Encoder + TFT + Cross Attention + Multi-task Head"""

    def __init__(self, config: HDPModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'auto' else 'cpu')

        self.static_encoder = XGBoostStaticEncoder(config.static_encoder_config)
        self.tft_encoder = TFTDynamicEncoder(config.tft_config)
        self._tft_backbone = None

        if config.use_attention:
            self.fusion = CrossAttentionFusion(config.fusion_config)
        else:
            self.fusion = SimpleFusion(config.static_embedding_dim, config.dynamic_embedding_dim, config.fusion_embedding_dim)

        if config.use_multitask:
            self.prediction_head = MultiTaskPredictionHead(config.multitask_config)
        else:
            self.prediction_head = SingleTaskHead(embedding_dim=config.fusion_embedding_dim)

        self.is_trained = False
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def forward(self, static_data: Optional[np.ndarray], dynamic_data: np.ndarray) -> Dict[str, Any]:
        """完整前向传播: Static Encoding → Dynamic Encoding → Fusion → Prediction"""
        E_s = None
        if self.config.use_static and static_data is not None:
            E_s = torch.FloatTensor(self.static_encoder.encode(static_data)).to(self.device)

        E_d = torch.FloatTensor(self.tft_encoder.encode(dynamic_data)).to(self.device)

        if self.config.use_static and E_s is not None and self.config.use_dynamic:
            Z = self.fusion(E_d, E_s)
        elif self.config.use_static and E_s is not None:
            Z = E_s
        else:
            Z = E_d

        if self.config.use_multitask:
            predictions = self.prediction_head(Z)
            P = torch.cat([predictions[name] for name in self.prediction_head.task_names], dim=-1)
        else:
            P = self.prediction_head(Z)
            predictions = {'single_head': P}

        result = {'E_s': E_s, 'E_d': E_d, 'Z': Z, 'predictions': predictions, 'P': P}
        if hasattr(self.fusion, 'get_attention_weights'):
            result['attention_weights'] = self.fusion.get_attention_weights()
        if hasattr(self.tft_encoder, 'get_variable_weights'):
            result['variable_weights'] = self.tft_encoder.get_variable_weights()
        return result

    def predict_kpi(self, static_data: np.ndarray, dynamic_data: np.ndarray) -> np.ndarray:
        """预测接口: 返回 [batch, 5] KPI 预测向量"""
        self.eval()
        with torch.no_grad():
            return self.forward(static_data, dynamic_data)['P'].cpu().numpy()

    def fit(
        self,
        static_train: np.ndarray,
        dynamic_train: np.ndarray,
        kpi_train: np.ndarray,
        static_val: Optional[np.ndarray] = None,
        dynamic_val: Optional[np.ndarray] = None,
        kpi_val: Optional[np.ndarray] = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """端到端训练: Phase 1 XGBoost → Phase 2 TFT → Phase 3 Fine-tune Fusion+Head"""
        # Step 1: 训练 Static Encoder
        if self.config.use_static:
            if verbose:
                print("\n  [Phase 1] 训练 Static Encoder (XGBoost)")
            self.static_encoder.fit(static_train, kpi_train)

        # Step 2: 训练 Dynamic Encoder (TFT)
        if verbose:
            print("\n  [Phase 2] 训练 Dynamic Encoder (TFT)")
        self.tft_encoder.fit(
            dynamic_train, kpi_train,
            dynamic_val, kpi_val,
            n_epochs=n_epochs
        )

        # Step 3: 端到端微调融合层 + 预测头
        if verbose:
            print("\n  [Phase 3] 端到端微调 Fusion + Prediction Head")

        # 获取嵌入
        with torch.no_grad():
            E_s_train = self.static_encoder.encode(static_train) if self.config.use_static else None
            E_d_train = self.tft_encoder.encode(dynamic_train)
            if dynamic_val is not None:
                E_s_val = self.static_encoder.encode(static_val) if (self.config.use_static and static_val is not None) else None
                E_d_val = self.tft_encoder.encode(dynamic_val)

        # 转移到 device
        E_d_t = torch.FloatTensor(E_d_train).to(self.device)
        kpi_t = torch.FloatTensor(kpi_train).to(self.device)
        if E_s_train is not None:
            E_s_t = torch.FloatTensor(E_s_train).to(self.device)
        else:
            E_s_t = None

        if dynamic_val is not None:
            E_d_val_t = torch.FloatTensor(E_d_val).to(self.device)
            kpi_val_t = torch.FloatTensor(kpi_val).to(self.device)
            if E_s_val is not None:
                E_s_val_t = torch.FloatTensor(E_s_val).to(self.device)
            else:
                E_s_val_t = None

        # 优化器
        fusion_params = list(self.fusion.parameters()) if hasattr(self, 'fusion') else []
        head_params = list(self.prediction_head.parameters())
        optimizer = torch.optim.Adam(fusion_params + head_params, lr=self.config.learning_rate)

        n_samples = len(E_d_t)
        best_val_loss = float('inf')
        no_improve = 0

        for epoch in range(n_epochs):
            self.train()

            # Mini-batch
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_E_d = E_d_t[start:end]
                batch_kpi = kpi_t[start:end]
                batch_E_s = E_s_t[start:end] if E_s_t is not None else None

                # Fusion
                if batch_E_s is not None and self.config.use_static:
                    Z = self.fusion(batch_E_d, batch_E_s)
                else:
                    Z = batch_E_d

                # Prediction
                if self.config.use_multitask:
                    predictions = self.prediction_head(Z)
                    loss, _ = self.prediction_head.compute_loss(predictions, batch_kpi)
                else:
                    pred = self.prediction_head(Z)
                    loss = self.prediction_head.compute_loss(pred, batch_kpi)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    fusion_params + head_params, 1.0
                )
                optimizer.step()

                epoch_loss += loss.item() * (end - start)

            epoch_loss /= n_samples
            self.train_loss_history.append(epoch_loss)

            # Validation
            if dynamic_val is not None:
                self.eval()
                with torch.no_grad():
                    if E_s_val_t is not None and self.config.use_static:
                        Z_val = self.fusion(E_d_val_t, E_s_val_t)
                    else:
                        Z_val = E_d_val_t

                    if self.config.use_multitask:
                        val_preds = self.prediction_head(Z_val)
                        val_loss, _ = self.prediction_head.compute_loss(val_preds, kpi_val_t)
                    else:
                        val_pred = self.prediction_head(Z_val)
                        val_loss = self.prediction_head.compute_loss(val_pred, kpi_val_t)

                self.val_loss_history.append(val_loss.item())

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                val_loss_str = "N/A"

            if verbose and epoch % 20 == 0:
                val_str = f", val_loss={val_loss.item():.6f}" if dynamic_val is not None else ""
                print(f"  [HDP] Epoch {epoch:3d}: train_loss={epoch_loss:.6f}{val_str}")

            if no_improve >= 20:
                if verbose:
                    print(f"  [HDP] Early stop @ epoch {epoch}")
                break

        self.is_trained = True
        if verbose:
            print(f"  [HDP] 端到端训练完成: {len(self.train_loss_history)} epochs")

    # ═══════════════════════════════════════════════════════════════
    #  保存 / 加载
    # ═══════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存完整模型"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存 PyTorch 部分
        torch.save({
            'fusion': self.fusion.state_dict(),
            'prediction_head': self.prediction_head.state_dict(),
            'config': self.config,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'is_trained': self.is_trained
        }, path + '.pt')

        # 保存 Static Encoder (XGBoost)
        self.static_encoder.save(path + '_static.pkl')

        # 保存 TFT Encoder
        self.tft_encoder.save(path + '_tft.pt')

        print(f"  [HDP] 模型已保存到 {path}*")

    def load(self, path: str) -> bool:
        """加载完整模型"""
        try:
            # 加载 PyTorch 部分
            checkpoint = torch.load(path + '.pt', map_location=self.device)
            self.config = checkpoint.get('config', self.config)

            # 重建模块
            if self.config.use_attention:
                self.fusion = CrossAttentionFusion(self.config.fusion_config)
            else:
                self.fusion = SimpleFusion(
                    self.config.static_embedding_dim,
                    self.config.dynamic_embedding_dim,
                    self.config.fusion_embedding_dim
                )
            if self.config.use_multitask:
                self.prediction_head = MultiTaskPredictionHead(self.config.multitask_config)
            else:
                self.prediction_head = SingleTaskHead(self.config.fusion_embedding_dim)

            self.fusion.load_state_dict(checkpoint['fusion'])
            self.prediction_head.load_state_dict(checkpoint['prediction_head'])
            self.train_loss_history = checkpoint.get('train_loss_history', [])
            self.val_loss_history = checkpoint.get('val_loss_history', [])
            self.is_trained = checkpoint.get('is_trained', False)

            # 加载 Static Encoder
            self.static_encoder.load(path + '_static.pkl')

            # 加载 TFT Encoder
            self.tft_encoder.load(path + '_tft.pt')

            self.to(self.device)
            self.eval()
            return True

        except Exception as e:
            print(f"  [HDP] 加载模型失败: {e}")
            return False

    def to(self, device):
        """移动模型到设备"""
        self.device = device
        if hasattr(self, 'fusion'):
            self.fusion = self.fusion.to(device)
        if hasattr(self, 'prediction_head'):
            self.prediction_head = self.prediction_head.to(device)
        return self
