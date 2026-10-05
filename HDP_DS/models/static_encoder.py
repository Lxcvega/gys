"""HDP-DS v3.1 Static Encoder — XGBoost 静态特征编码器"""

import numpy as np
import xgboost as xgb
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StaticEncoderConfig:
    """静态编码器配置"""
    static_dim: int = 6
    embedding_dim: int = 32
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    use_leaf_embedding: bool = True
    verbose: bool = True
    feature_names: List[str] = field(default_factory=lambda: [
        'enterprise_scale', 'registered_capital', 'iso_certified',
        'env_certified', 'credit_rating', 'enterprise_age'
    ])


class XGBoostStaticEncoder:
    """将 6 维静态特征编码为 d_s 维嵌入向量 E_s"""

    def __init__(self, config: Optional[StaticEncoderConfig] = None):
        self.config = config or StaticEncoderConfig()
        self.model: Optional[xgb.XGBRegressor] = None
        self.is_fitted = False
        self.feature_importances_: Optional[np.ndarray] = None
        self.projection_weight: Optional[np.ndarray] = None
        self.projection_bias: Optional[np.ndarray] = None

    def fit(self, static_data: np.ndarray, kpi_targets: np.ndarray):
        """训练 XGBoost 编码器"""
        n_samples = static_data.shape[0]
        if n_samples < 10:
            print(f"  [XGBoost] 数据不足 ({n_samples} < 10)，使用随机投影")
            self._init_random_projection()
            self.is_fitted = True
            return

        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators, max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate, subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree, reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda, random_state=self.config.random_state,
            verbosity=0 if not self.config.verbose else 1,
            objective='reg:squarederror', n_jobs=-1
        )

        from sklearn.multioutput import MultiOutputRegressor
        multi_model = MultiOutputRegressor(self.model, n_jobs=-1)
        multi_model.fit(static_data, kpi_targets)

        if hasattr(multi_model.estimators_[0], 'feature_importances_'):
            self.feature_importances_ = multi_model.estimators_[0].feature_importances_
        else:
            self.feature_importances_ = np.ones(self.config.static_dim) / self.config.static_dim

        self._init_projection_layer(static_data)
        self.is_fitted = True

        if self.config.verbose:
            print(f"  [StaticEncoder] XGBoost 训练完成: {self.config.n_estimators} trees, "
                  f"embedding_dim={self.config.embedding_dim}")
            self._print_feature_importance()

    def encode(self, static_data: np.ndarray) -> np.ndarray:
        """编码静态特征为嵌入向量"""
        if static_data.ndim == 1:
            static_data = static_data.reshape(1, -1)
        if not self.is_fitted:
            raise ValueError("StaticEncoder 尚未训练，请先调用 fit()")

        if self.model is not None:
            try:
                xgb_pred = self.model.predict(static_data)
                if xgb_pred.ndim == 1:
                    xgb_pred = xgb_pred.reshape(-1, 1)
                if xgb_pred.shape[1] < self.config.embedding_dim:
                    repeats = self.config.embedding_dim // xgb_pred.shape[1] + 1
                    xgb_pred = np.tile(xgb_pred, (1, repeats))[:, :self.config.embedding_dim]
                return xgb_pred.astype(np.float32)
            except Exception:
                pass

        if self.projection_weight is not None:
            return static_data @ self.projection_weight.T + self.projection_bias
        return np.zeros((static_data.shape[0], self.config.embedding_dim), dtype=np.float32)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.feature_importances_ is None:
            return {name: 1.0 / len(self.config.feature_names) for name in self.config.feature_names}
        importance_dict = {}
        for i, name in enumerate(self.config.feature_names):
            if i < len(self.feature_importances_):
                importance_dict[name] = float(self.feature_importances_[i])
        total = sum(importance_dict.values())
        if total > 0:
            for k in importance_dict:
                importance_dict[k] /= total
        return importance_dict

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config, 'model': self.model,
                'projection_weight': self.projection_weight,
                'projection_bias': self.projection_bias,
                'feature_importances_': self.feature_importances_,
                'is_fitted': self.is_fitted
            }, f)
        if self.config.verbose:
            print(f"  [StaticEncoder] 已保存到 {path}")

    def load(self, path: str) -> bool:
        """加载模型"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.config = data.get('config', self.config)
            self.model = data.get('model')
            self.projection_weight = data.get('projection_weight')
            self.projection_bias = data.get('projection_bias')
            self.feature_importances_ = data.get('feature_importances_')
            self.is_fitted = data.get('is_fitted', False)
            return True
        except Exception as e:
            print(f"  [StaticEncoder] 加载失败: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════
    #  内部方法
    # ═══════════════════════════════════════════════════════════════

    def _init_projection_layer(self, static_data: np.ndarray):
        """初始化投影层"""
        input_dim = static_data.shape[1]
        output_dim = self.config.embedding_dim
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.projection_weight = np.random.randn(output_dim, input_dim).astype(np.float32) * scale
        self.projection_bias = np.zeros(output_dim, dtype=np.float32)

    def _init_random_projection(self):
        """当数据不足时使用随机投影"""
        input_dim = self.config.static_dim
        output_dim = self.config.embedding_dim
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.projection_weight = np.random.randn(output_dim, input_dim).astype(np.float32) * scale
        self.projection_bias = np.zeros(output_dim, dtype=np.float32)
        self.feature_importances_ = np.ones(input_dim) / input_dim

    def _print_feature_importance(self):
        """打印特征重要性"""
        importance = self.get_feature_importance()
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("  [Feature Importance]")
        for name, val in sorted_items:
            bar = '█' * int(val * 50)
            print(f"    {name:25s}: {val:.4f} {bar}")
