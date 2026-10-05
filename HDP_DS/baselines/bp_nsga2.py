"""HDP-DS v3.1 Baseline — 原 BP + NSGA-II 对比基线"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class BPNSGA2Config:
    """BP + NSGA-II Baseline 配置"""
    # BP 参数
    bp_hidden_layers: tuple = (64, 32)
    bp_activation: str = 'relu'
    bp_learning_rate: float = 0.001
    bp_max_iter: int = 2000
    bp_alpha: float = 0.01

    # NSGA-II 参数
    nsga2_population_size: int = 100
    nsga2_generations: int = 50
    nsga2_crossover_rate: float = 0.9
    nsga2_mutation_rate: float = 0.02

    # 目标权重
    objective_weights: Optional[Dict[str, float]] = None

    # 随机种子
    random_state: int = 42

    def __post_init__(self):
        if self.objective_weights is None:
            self.objective_weights = {
                'cost': 0.30,
                'quality': 0.25,
                'delivery': 0.20,
                'environmental': 0.15,
                'reliability': 0.10
            }


@dataclass
class BaselineResult:
    """Baseline 结果"""
    supplier_id: str
    bp_score: float                    # BP 预测评分
    objectives: Dict[str, float]      # NSGA-II 目标值
    pareto_rank: int                   # Pareto 等级
    final_score: float                 # 综合评分


class BPNSGA2Baseline:
    """
    BP + NSGA-II Baseline

    保留原始 BP 神经网络预测 + NSGA-II 多目标优化。
    用于与 HDP-DS 的对比实验。
    """

    def __init__(self, config: Optional[BPNSGA2Config] = None):
        self.config = config or BPNSGA2Config()
        self.bp_model: Optional[MLPRegressor] = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

        # 评估指标
        self.r2_train: Optional[float] = None
        self.r2_test: Optional[float] = None
        self.rmse_test: Optional[float] = None
        self.mae_test: Optional[float] = None

    # ═══════════════════════════════════════════════════════════════
    #  训练
    # ═══════════════════════════════════════════════════════════════

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练 BP 神经网络

        Args:
            X: [n_samples, 10] 供应商特征
            y: [n_samples] 目标评分 (0-100)
        """
        if not SKLEARN_AVAILABLE:
            print("  [Baseline] sklearn 未安装")
            return

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 切分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=self.config.random_state
        )

        # 训练 BP
        self.bp_model = MLPRegressor(
            hidden_layer_sizes=self.config.bp_hidden_layers,
            activation=self.config.bp_activation,  # type: ignore[arg-type]
            solver='adam',
            alpha=self.config.bp_alpha,
            learning_rate='adaptive',
            learning_rate_init=self.config.bp_learning_rate,
            max_iter=self.config.bp_max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.bp_model.fit(X_train, y_train)

        # 评估
        train_pred = self.bp_model.predict(X_train)
        test_pred = self.bp_model.predict(X_test)

        self.r2_train = r2_score(y_train, train_pred)
        self.r2_test = r2_score(y_test, test_pred)
        self.rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
        self.mae_test = mean_absolute_error(y_test, test_pred)
        self.is_trained = True

        print(f"  [Baseline] BP 训练完成:")
        print(f"    R² 训练集: {self.r2_train:.4f}")
        print(f"    R² 测试集: {self.r2_test:.4f}")
        print(f"    RMSE:     {self.rmse_test:.2f}")
        print(f"    MAE:      {self.mae_test:.2f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """BP 预测"""
        if self.bp_model is None:
            raise ValueError("BP 模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.bp_model.predict(X_scaled)

    # ═══════════════════════════════════════════════════════════════
    #  NSGA-II (简化版)
    # ═══════════════════════════════════════════════════════════════

    def nsga2_optimize(
        self,
        feature_matrix: np.ndarray,     # [n_suppliers, 10]
        supplier_ids: List[str]
    ) -> List[BaselineResult]:
        """
        NSGA-II 优化 (原逻辑保留)

        Args:
            feature_matrix: 供应商特征矩阵
            supplier_ids: 供应商 ID 列表
        """
        n_suppliers = feature_matrix.shape[0]

        # BP 预测评分
        bp_scores = self.predict(feature_matrix)

        results = []
        for i in range(n_suppliers):
            features = feature_matrix[i]

            # 计算 5 目标 (沿用原始逻辑)
            objectives = {
                'cost': features[0] / (features[0].max() + 1e-8) if isinstance(features, np.ndarray) else 0.5,
                'quality': features[3] / 100.0,
                'delivery': 1.0 - features[4] / 100.0,
                'environmental': features[8] / 5.0,
                'reliability': bp_scores[i] / 100.0
            }

            # 综合评分 (加权和)
            assert self.config.objective_weights is not None
            final_score = sum(
                objectives[k] * self.config.objective_weights[k]
                for k in self.config.objective_weights
            )

            results.append(BaselineResult(
                supplier_id=supplier_ids[i],
                bp_score=bp_scores[i],
                objectives=objectives,
                pareto_rank=0,
                final_score=final_score
            ))

        # 按综合评分排序
        results.sort(key=lambda r: r.final_score, reverse=True)

        # Pareto rank (简化: 按 final_score 排序后分配 rank)
        for rank, r in enumerate(results):
            r.pareto_rank = rank

        return results

    # ═══════════════════════════════════════════════════════════════
    #  对比接口
    # ═══════════════════════════════════════════════════════════════

    def compare_with_hdp(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hdp_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        与 HDP-DS 对比

        Returns:
            {metric_name: value}
        """
        if not self.is_trained:
            return {'error': 'BP 未训练'}

        bp_pred = self.predict(X_test)

        comparison = {
            'BP_R2': r2_score(y_test, bp_pred),
            'BP_RMSE': np.sqrt(mean_squared_error(y_test, bp_pred)),
            'BP_MAE': mean_absolute_error(y_test, bp_pred),
        }

        if hdp_predictions is not None:
            comparison['HDP_R2'] = r2_score(y_test, hdp_predictions)
            comparison['HDP_RMSE'] = np.sqrt(mean_squared_error(y_test, hdp_predictions))
            comparison['HDP_MAE'] = mean_absolute_error(y_test, hdp_predictions)
            comparison['R2_Improvement'] = comparison['HDP_R2'] - comparison['BP_R2']

        return comparison

    def print_comparison(self, comparison: Dict[str, Any]):
        """打印对比结果"""
        print("\n" + "=" * 55)
        print("  Baseline vs HDP-DS 对比")
        print("=" * 55)

        if 'BP_R2' in comparison:
            print(f"\n  BP 神经网络:")
            print(f"    R²:   {comparison['BP_R2']:.4f}")
            print(f"    RMSE: {comparison['BP_RMSE']:.2f}")
            print(f"    MAE:  {comparison['BP_MAE']:.2f}")

        if 'HDP_R2' in comparison:
            print(f"\n  HDP-DS:")
            print(f"    R²:   {comparison['HDP_R2']:.4f}")
            print(f"    RMSE: {comparison['HDP_RMSE']:.2f}")
            print(f"    MAE:  {comparison['HDP_MAE']:.2f}")

        if 'R2_Improvement' in comparison:
            impr = comparison['R2_Improvement']
            print(f"\n  R² 提升: {impr:+.4f} ({impr * 100:+.2f}%)")

        print("=" * 55)
