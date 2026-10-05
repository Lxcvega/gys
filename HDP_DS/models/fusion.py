"""HDP-DS v3.1 Fusion Layer — Cross Attention 融合模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """融合层配置"""
    static_dim: int = 32
    dynamic_dim: int = 32
    embedding_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True


class CrossAttentionFusion(nn.Module):
    """Cross Attention 融合层: 动态嵌入为 Query, 静态嵌入为 Key/Value"""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        assert config.dynamic_dim % config.num_heads == 0, \
            f"dynamic_dim ({config.dynamic_dim}) 必须能被 num_heads ({config.num_heads}) 整除"

        self.head_dim = config.dynamic_dim // config.num_heads
        self.q_proj = nn.Linear(config.dynamic_dim, config.dynamic_dim)
        self.k_proj = nn.Linear(config.static_dim, config.dynamic_dim)
        self.v_proj = nn.Linear(config.static_dim, config.dynamic_dim)
        self.out_proj = nn.Linear(config.dynamic_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_layer_norm:
            self.norm_q = nn.LayerNorm(config.dynamic_dim)
            self.norm_k = nn.LayerNorm(config.static_dim)
            self.norm_out = nn.LayerNorm(config.embedding_dim)
        else:
            self.norm_q = self.norm_k = self.norm_out = nn.Identity()

        if config.use_residual and config.dynamic_dim != config.embedding_dim:
            self.residual_proj = nn.Linear(config.dynamic_dim, config.embedding_dim)
        else:
            self.residual_proj = nn.Identity() if config.dynamic_dim == config.embedding_dim else nn.Linear(config.dynamic_dim, config.embedding_dim)

        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, E_d: torch.Tensor, E_s: torch.Tensor) -> torch.Tensor:
        """E_d: [batch, dynamic_dim], E_s: [batch, static_dim] → Z: [batch, embedding_dim]"""
        batch_size = E_d.size(0)
        E_d, E_s = self.norm_q(E_d), self.norm_k(E_s)
        Q, K, V = self.q_proj(E_d), self.k_proj(E_s), self.v_proj(E_s)

        Q = Q.view(batch_size, self.config.num_heads, self.head_dim).unsqueeze(2)
        K = K.view(batch_size, self.config.num_heads, self.head_dim).unsqueeze(2)
        V = V.view(batch_size, self.config.num_heads, self.head_dim).unsqueeze(2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        self.last_attention_weights = attn_weights.detach()

        context = torch.matmul(attn_weights, V).squeeze(2).reshape(batch_size, -1)
        Z = self.out_proj(context)
        if self.config.use_residual:
            Z = Z + self.residual_proj(E_d)
        return self.norm_out(Z)

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """获取注意力权重"""
        return self.last_attention_weights.cpu().numpy() if self.last_attention_weights is not None else None


class SimpleFusion(nn.Module):
    """简单融合 (用于消融实验 — use_attention=False)"""

    def __init__(self, static_dim: int, dynamic_dim: int, embedding_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, E_d: torch.Tensor, E_s: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([E_d, E_s], dim=-1))
