"""HDP-DS v3.1 评估指标模块: MAE, RMSE, MAPE, R², Kendall Tau"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MetricResult:
    """单指标结果"""
    name: str
    value: float
    per_task: Optional[Dict[str, float]] = None


class EvaluationMetrics:
    """评估指标计算器: MAE, RMSE, MAPE, R²"""

    def __init__(self):
        self.task_names = [
            'Delivery Reliability',
            'Quality Stability',
            'Environmental Risk',
            'Cost Stability',
            'Service Capability'
        ]

    def compute_all(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
        计算所有指标

        Args:
            y_true: [n_samples, 5] 真实 KPI
            y_pred: [n_samples, 5] 预测 KPI

        Returns:
            {metric_name: MetricResult}
        """
        results = {}

        # MAE
        results['MAE'] = self.mae(y_true, y_pred)

        # RMSE
        results['RMSE'] = self.rmse(y_true, y_pred)

        # MAPE
        results['MAPE'] = self.mape(y_true, y_pred)

        # R²
        results['R2'] = self.r2(y_true, y_pred)

        return results

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
        """Mean Absolute Error"""
        errors = np.abs(y_true - y_pred)
        overall = float(np.mean(errors))
        per_task = {
            name: float(np.mean(errors[:, i]))
            for i, name in enumerate(self.task_names)
        }
        return MetricResult(name='MAE', value=overall, per_task=per_task)

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
        """Root Mean Squared Error"""
        errors = (y_true - y_pred) ** 2
        overall = float(np.sqrt(np.mean(errors)))
        per_task = {
            name: float(np.sqrt(np.mean(errors[:, i])))
            for i, name in enumerate(self.task_names)
        }
        return MetricResult(name='RMSE', value=overall, per_task=per_task)

    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
        """Mean Absolute Percentage Error"""
        # 避免除零
        denominator = np.abs(y_true) + 1e-8
        errors = np.abs((y_true - y_pred) / denominator)
        overall = float(np.mean(errors) * 100)  # 转换为百分比
        per_task = {
            name: float(np.mean(errors[:, i]) * 100)
            for i, name in enumerate(self.task_names)
        }
        return MetricResult(name='MAPE', value=overall, per_task=per_task)

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
        """R² Coefficient of Determination"""
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        r2_per_task = 1 - ss_res / (ss_tot + 1e-8)
        overall = float(np.mean(r2_per_task))
        per_task = {
            name: float(r2_per_task[i])
            for i, name in enumerate(self.task_names)
        }
        return MetricResult(name='R2', value=overall, per_task=per_task)

    def kendall_tau(
        self, rankings_true: List[int], rankings_pred: List[int]
    ) -> float:
        """
        Kendall Tau 排序一致性

        Args:
            rankings_true: 真实排序 (1-based)
            rankings_pred: 预测排序 (1-based)

        Returns:
            tau: [-1, 1], 1 表示完全一致
        """
        tau, p_value = kendalltau(rankings_true, rankings_pred)
        return float(tau)

    @staticmethod
    def print_metrics(results: Dict[str, MetricResult], title: str = "评估结果"):
        """打印指标报告"""
        print(f"\n{'=' * 55}")
        print(f"  {title}")
        print(f"{'=' * 55}")

        for metric_name, metric_result in results.items():
            print(f"\n  {metric_name}: {metric_result.value:.4f}")
            if metric_result.per_task:
                for task_name, task_value in metric_result.per_task.items():
                    print(f"    {task_name}: {task_value:.4f}")

        print(f"{'=' * 55}")


# ─── 实验对比工具 ───────────────────────────────────────────────

class ExperimentComparator:
    """
    模型对比工具

    比较多个模型的预测指标并生成对比表。
    """

    def __init__(self):
        self.metrics = EvaluationMetrics()

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        比较多个模型的预测

        Args:
            y_true: [n, 5] 真实值
            predictions: {model_name: [n, 5]}

        Returns:
            {model_name: {metric: value}}
        """
        comparison = {}
        for model_name, y_pred in predictions.items():
            results = self.metrics.compute_all(y_true, y_pred)
            comparison[model_name] = {
                name: result.value for name, result in results.items()
            }
        return comparison

    def print_comparison_table(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ):
        """打印对比表"""
        comparison = self.compare_models(y_true, predictions)

        print(f"\n{'=' * 70}")
        print(f"  模型对比实验")
        print(f"{'=' * 70}")

        # 表头
        model_names = list(comparison.keys())
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        header = f"  {'模型':20s}" + "".join([f"{m:>10s}" for m in metrics])
        print(f"  {'─' * 70}")
        print(header)
        print(f"  {'─' * 70}")

        for model_name in model_names:
            row = f"  {model_name:20s}"
            for m in metrics:
                val = comparison[model_name].get(m, float('nan'))
                row += f"{val:>10.4f}"
            print(row)

        print(f"  {'─' * 70}")
