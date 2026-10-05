"""HDP-DS v3.1 Dynamic Encoder — Temporal Fusion Transformer (TFT)"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TFTConfig:
    """TFT 编码器配置"""
    dynamic_dim: int = 6
    embedding_dim: int = 32
    forecast_horizon: int = 3
    hidden_size: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 10
    device: str = 'auto'
    random_state: int = 42
    use_variable_selection: bool = True
    variable_dim: int = 8


class VariableSelectionNetwork(nn.Module):
    """变量选择网络: 为每个时间步的每个变量计算软权重"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, input_dim), nn.Softmax(dim=-1)
        )
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weight_network(x)
        features = self.feature_network(x)
        output = weights * features
        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """可解释多头注意力: 将多头输出平均合并为单头"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, query, key, value, mask=None):
        """query, key, value: [batch, T, d_model] → [batch, T_q, d_model]"""
        batch_size = query.size(0)
        Q, K, V = self.q_linear(query), self.k_linear(key), self.v_linear(value)

        # 分头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 保存注意力权重 (平均所有头)
        self.attention_weights = attn_weights.mean(dim=1)  # [batch, T_q, T_k]

        # 加权求和
        context = torch.matmul(attn_weights, V)  # [batch, h, T_q, d_k]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.out_linear(context)
        return output


class GatedResidualNetwork(nn.Module):
    """门控残差网络 (GRN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        # 残差连接
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T, input_dim]

        Returns:
            [batch, T, output_dim]
        """
        residual = self.skip(x)

        hidden = self.fc1(x)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        gate = self.gate(hidden)
        output = gate * hidden + (1 - gate) * residual

        return output


class TFTBackbone(nn.Module):
    """
    TFT 骨干网络 (精简版)

    包含:
      1. Variable Selection Network
      2. LSTM Encoder
      3. Interpretable Multi-Head Attention
      4. Gated Residual Network
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config

        # 变量选择网络
        if config.use_variable_selection:
            self.vsn = VariableSelectionNetwork(
                input_dim=config.dynamic_dim,
                hidden_dim=config.variable_dim,
                output_dim=config.dynamic_dim
            )
        else:
            self.vsn = None

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=config.dynamic_dim,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # 注意力层
        self.attention = InterpretableMultiHeadAttention(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout
        )

        # 门控残差网络
        self.grn = GatedResidualNetwork(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size,
            dropout=config.dropout
        )

        # 输出投影到 embedding_dim
        self.output_projection = nn.Linear(config.hidden_size, config.embedding_dim)

        # 变量选择权重缓存 (用于可解释性)
        self.last_vsn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T, dynamic_dim]

        Returns:
            E_d: [batch, embedding_dim] 动态嵌入
        """
        batch_size, T, _ = x.shape

        # 1. 变量选择
        if self.vsn is not None:
            x, vsn_weights = self.vsn(x)
            self.last_vsn_weights = vsn_weights.detach()
        else:
            self.last_vsn_weights = None

        # 2. LSTM 编码
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: [batch, T, hidden_size]

        # 3. 自注意力 (使用 LSTM 输出作为 Q, K, V)
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)  # [batch, T, hidden_size]

        # 4. 门控残差
        grn_out = self.grn(attn_out + lstm_out)  # [batch, T, hidden_size]

        # 5. 取最后一个时间步作为序列表示
        last_step = grn_out[:, -1, :]  # [batch, hidden_size]

        # 6. 投影到嵌入空间
        E_d = self.output_projection(last_step)  # [batch, embedding_dim]

        return E_d

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重 (可解释性)"""
        return self.attention.attention_weights

    def get_variable_weights(self) -> Optional[torch.Tensor]:
        """获取变量选择权重 (可解释性)"""
        return self.last_vsn_weights


class TFTDynamicEncoder:
    """
    TFT 动态编码器 (高级封装)

    包装 TFTBackbone，提供训练/编码/保存/加载接口。
    当 pytorch-forecasting 可用时，优先使用官方实现。
    """

    def __init__(self, config: Optional[TFTConfig] = None):
        self.config = config or TFTConfig()

        # 自动选择设备
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        # 模型
        self.model: Optional[TFTBackbone] = None
        self.is_trained = False

        # 损失历史
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    # ═══════════════════════════════════════════════════════════════
    #  训练接口
    # ═══════════════════════════════════════════════════════════════

    def fit(
        self,
        dynamic_train: np.ndarray,     # [n_train, T, 6]
        kpi_train: np.ndarray,         # [n_train, 5]
        dynamic_val: Optional[np.ndarray] = None,
        kpi_val: Optional[np.ndarray] = None,
        n_epochs: Optional[int] = None
    ):
        """
        训练 TFT 编码器

        使用多任务预测作为代理任务来学习动态嵌入。
        """
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        # 初始化模型
        self.model = TFTBackbone(self.config).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # 将数据转换为 tensor
        X_train = torch.FloatTensor(dynamic_train).to(self.device)
        y_train = torch.FloatTensor(kpi_train).to(self.device)

        has_val = dynamic_val is not None and kpi_val is not None
        if has_val:
            X_val = torch.FloatTensor(dynamic_val).to(self.device)
            y_val = torch.FloatTensor(kpi_val).to(self.device)

        n_epochs = n_epochs or self.config.max_epochs
        best_val_loss = float('inf')
        patience_counter = 0

        # 简单的预测头 (用于代理训练)
        pred_head = nn.Linear(self.config.embedding_dim, 5).to(self.device)
        pred_optimizer = torch.optim.Adam(pred_head.parameters(), lr=self.config.learning_rate)
        _n_epochs = self.config.max_epochs if n_epochs is None else n_epochs

        for epoch in range(_n_epochs):
            self.model.train()
            optimizer.zero_grad()
            pred_optimizer.zero_grad()
            E_d = self.model(X_train)
            pred = pred_head(E_d)
            loss = F.mse_loss(pred, y_train)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            pred_optimizer.step()

            self.train_loss_history.append(loss.item())

            # 验证
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    E_d_val = self.model(X_val)
                    pred_val = pred_head(E_d_val)
                    val_loss = F.mse_loss(pred_val, y_val)
                self.val_loss_history.append(val_loss.item())

                # Early stopping
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        if self.config.embedding_dim:
                            print(f"  [TFT] Early stop @ epoch {epoch}")
                        break
            else:
                if epoch % 10 == 0 and epoch > 0:
                    pass  # 静默训练

            if epoch % 20 == 0 and self.config.embedding_dim:
                val_str = f", val_loss={val_loss.item():.6f}" if has_val else ""
                print(f"  [TFT] Epoch {epoch:3d}: train_loss={loss.item():.6f}{val_str}")

        self.is_trained = True
        if self.config.embedding_dim:
            print(f"  [TFT] 训练完成: {len(self.train_loss_history)} epochs, "
                  f"final_loss={self.train_loss_history[-1]:.6f}")

        # 移除代理预测头，保留特征提取能力
        self.model.eval()

    def fine_tune(
        self,
        dynamic_new: np.ndarray,
        kpi_new: np.ndarray,
        n_epochs: int = 10,
        lr: float = 1e-4
    ):
        """
        微调 TFT (用于反馈闭环)

        Args:
            dynamic_new: 新动态数据 [n, T, 6]
            kpi_new: 新 KPI 标签 [n, 5]
            n_epochs: 微调轮数
            lr: 学习率 (比初始训练小)
        """
        if not self.is_trained or self.model is None:
            print("  [TFT] 模型未训练，跳过微调")
            return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        X_new = torch.FloatTensor(dynamic_new).to(self.device)
        y_new = torch.FloatTensor(kpi_new).to(self.device)

        pred_head = nn.Linear(self.config.embedding_dim, 5).to(self.device)
        pred_optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr)
        _n_epochs = 10 if n_epochs is None else n_epochs

        for epoch in range(_n_epochs):
            optimizer.zero_grad()
            pred_optimizer.zero_grad()
            E_d = self.model(X_new)
            pred = pred_head(E_d)
            loss = F.mse_loss(pred, y_new)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()
            pred_optimizer.step()

        self.model.eval()
        print(f"  [TFT] 微调完成: {_n_epochs} epochs, loss={loss.item():.6f}")

    def encode(self, dynamic_data: np.ndarray) -> np.ndarray:
        """
        编码动态数据为嵌入向量

        Args:
            dynamic_data: [batch, T, 6] 或 [T, 6]

        Returns:
            E_d: [batch, embedding_dim]
        """
        if dynamic_data.ndim == 2:
            dynamic_data = dynamic_data[np.newaxis, ...]

        if not self.is_trained or self.model is None:
            raise ValueError("TFT 尚未训练，请先调用 fit()")

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(dynamic_data).to(self.device)
            E_d = self.model(x)

        return E_d.cpu().numpy()

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """获取注意力权重 (可解释性)"""
        if self.model is None:
            return None
        weights = self.model.get_attention_weights()
        if weights is None:
            return None
        return weights.cpu().numpy()

    def get_variable_weights(self) -> Optional[np.ndarray]:
        """获取变量选择权重 (可解释性)"""
        if self.model is None:
            return None
        weights = self.model.get_variable_weights()
        if weights is None:
            return None
        return weights.cpu().numpy()

    # ═══════════════════════════════════════════════════════════════
    #  保存 / 加载
    # ═══════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            print("  [TFT] 无模型可保存")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'config': self.config,
            'model_state': self.model.state_dict(),
            'is_trained': self.is_trained,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history
        }
        torch.save(save_dict, path)
        print(f"  [TFT] 已保存到 {path}")

    def load(self, path: str) -> bool:
        """加载模型"""
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.config = checkpoint.get('config', self.config)
            self.model = TFTBackbone(self.config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.is_trained = checkpoint.get('is_trained', False)
            self.train_loss_history = checkpoint.get('train_loss_history', [])
            self.val_loss_history = checkpoint.get('val_loss_history', [])
            self.model.eval()
            return True
        except Exception as e:
            print(f"  [TFT] 加载失败: {e}")
            return False
