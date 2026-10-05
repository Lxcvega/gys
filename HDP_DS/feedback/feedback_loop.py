"""HDP-DS v3.1 闭环反馈模块: 预测→选择→反馈→更新的闭环更新机制"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeedbackConfig:
    """反馈闭环配置"""
    n_cycles: int = 5
    feedback_noise_std: float = 0.05
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 1e-4
    retrain_tft: bool = True
    verbose: bool = True
    simulate_drift: bool = True
    drift_rate: float = 0.01


@dataclass
class CycleResult:
    """单个周期结果"""
    cycle: int
    predictions: np.ndarray               # [n_suppliers, 5] 预测 KPI
    rankings: List[Any]                   # TOPSIS 排序结果
    real_kpi: Optional[np.ndarray] = None # 真实 KPI
    feedback_applied: bool = False        # 是否应用了反馈


class FeedbackLoop:
    """闭环反馈机制: 多周期"预测→选择→反馈→更新"闭环"""

    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
        model=None,            # HDPModel
        optimizer=None,        # HDPNSGA2
        decision_maker=None,   # TOPSISDecision
        data_loader=None       # HDPDataLoader
    ):
        self.config = config or FeedbackConfig()
        self.model = model
        self.optimizer = optimizer
        self.decision_maker = decision_maker
        self.data_loader = data_loader

        # 历史记录
        self.cycle_history: List[CycleResult] = []
        self.kpi_history: List[np.ndarray] = []  # 所有历史 KPI 数据
        self.dynamic_history: List[np.ndarray] = []  # 所有历史动态数据

    # ═══════════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════════

    def run_cycles(
        self,
        static_data: np.ndarray,       # [n_suppliers, static_dim]
        dynamic_data: np.ndarray,      # [n_suppliers, T, 6]
        initial_kpi: np.ndarray,       # [n_suppliers, 5]
        supplier_ids: List[str],
        supplier_names: Optional[List[str]] = None,
        real_kpi_callback: Optional[Callable] = None
    ) -> List[CycleResult]:
        """
        运行多周期闭环

        Args:
            static_data: 静态特征
            dynamic_data: 动态时序
            initial_kpi: 初始 KPI
            supplier_ids: 供应商 ID 列表
            real_kpi_callback: 获取真实 KPI 的回调函数

        Returns:
            cycle_history: 各周期结果
        """
        if supplier_names is None:
            supplier_names = supplier_ids

        current_dynamic = dynamic_data.copy()
        current_kpi = initial_kpi.copy()

        for cycle in range(1, self.config.n_cycles + 1):
            if self.config.verbose:
                print(f"\n  {'=' * 50}")
                print(f"  Feedback Cycle {cycle}/{self.config.n_cycles}")
                print(f"  {'=' * 50}")

            # ── Step 1: 预测 KPI ──
            if self.model is not None:
                predictions = self.model.predict_kpi(static_data, current_dynamic)
            else:
                predictions = current_kpi.copy()

            # ── Step 2: NSGA-II + TOPSIS ──
            if self.optimizer is not None and self.decision_maker is not None:
                pareto_solutions, pareto_front = self.optimizer.optimize(
                    predictions, supplier_ids
                )

                if len(pareto_front) > 0:
                    rankings = self.decision_maker.rank(
                        pareto_front, supplier_ids, supplier_names, predictions
                    )
                else:
                    # Fallback: 直接对 KPI 排序
                    rankings = self.decision_maker.rank_suppliers_from_kpi(
                        predictions, supplier_ids, supplier_names
                    )
            else:
                rankings = []

            # ── Step 3: 获取真实 KPI (仿真或回调) ──
            if real_kpi_callback is not None:
                real_kpi = real_kpi_callback(cycle, predictions, rankings)
            elif self.config.simulate_drift:
                # 模拟 KPI 漂移 + 噪声
                drift = self.config.drift_rate * cycle
                noise = np.random.randn(*predictions.shape) * self.config.feedback_noise_std
                real_kpi = np.clip(predictions + drift + noise, 0, 1)
            else:
                real_kpi = predictions.copy()

            # ── Step 4: 追加到历史数据 ──
            self.kpi_history.append(real_kpi)
            self.dynamic_history.append(current_dynamic)

            # ── Step 5: Fine-tune TFT ──
            if self.config.retrain_tft and self.model is not None:
                self._fine_tune(real_kpi)
                feedback_applied = True
            else:
                feedback_applied = False

            # ── Step 6: 更新动态数据 (追加未来步) ──
            current_dynamic = self._update_dynamic_data(
                current_dynamic, real_kpi
            )
            current_kpi = real_kpi

            # 记录周期结果
            cycle_result = CycleResult(
                cycle=cycle,
                predictions=predictions,
                rankings=rankings,
                real_kpi=real_kpi,
                feedback_applied=feedback_applied
            )
            self.cycle_history.append(cycle_result)

            if self.config.verbose:
                top = rankings[0] if rankings else None
                if top:
                    print(f"  → 推荐: #{top.rank} {top.supplier_name} "
                          f"(贴近度={top.closeness:.4f})")
                print(f"  → KPI 变化: {real_kpi.mean():.4f} "
                      f"(vs 预测 {predictions.mean():.4f})")

        return self.cycle_history

    # ═══════════════════════════════════════════════════════════════
    #  内部方法
    # ═══════════════════════════════════════════════════════════════

    def _fine_tune(self, real_kpi: np.ndarray):
        """微调 TFT Encoder"""
        if self.model is None:
            return

        # 使用最新的动态数据 + 真实 KPI 微调
        latest_dynamic = self.dynamic_history[-1]

        if hasattr(self.model.tft_encoder, 'fine_tune'):
            self.model.tft_encoder.fine_tune(
                latest_dynamic, real_kpi,
                n_epochs=self.config.fine_tune_epochs,
                lr=self.config.fine_tune_lr
            )

    def _update_dynamic_data(
        self, current_dynamic: np.ndarray, real_kpi: np.ndarray
    ) -> np.ndarray:
        """
        更新动态数据

        将真实 KPI 转换为动态特征并追加到时序末尾。
        """
        n_suppliers, T, N = current_dynamic.shape

        # KPI → 动态特征转换 (5维 → 6维)
        kpi_to_dynamic = np.zeros((n_suppliers, 1, N))
        if real_kpi.shape[1] >= 5:
            # 近似映射: KPI → 动态特征
            kpi_to_dynamic[:, 0, 0] = real_kpi[:, 0] * 100  # delivery_rate
            kpi_to_dynamic[:, 0, 1] = real_kpi[:, 1] * 100  # quality_rate
            kpi_to_dynamic[:, 0, 2] = (1 - real_kpi[:, 2]) * 10  # complaint_count
            kpi_to_dynamic[:, 0, 3] = (1 - real_kpi[:, 2]) * 5   # env_incident
            kpi_to_dynamic[:, 0, 4] = (1 - real_kpi[:, 3]) * 10  # cost_change
            kpi_to_dynamic[:, 0, 5] = (1 - real_kpi[:, 4]) * 24  # service_time

        # 拼接: [n, T+1, N]
        updated = np.concatenate([current_dynamic, kpi_to_dynamic], axis=1)

        # 如果超出最大长度，滑动窗口截断
        max_T = T + 3  # 允许额外 3 个月的扩展
        if updated.shape[1] > max_T:
            updated = updated[:, -max_T:, :]

        return updated

    # ═══════════════════════════════════════════════════════════════
    #  报告
    # ═══════════════════════════════════════════════════════════════

    def generate_report(self) -> str:
        """生成闭环报告"""
        if not self.cycle_history:
            return "无闭环数据"

        lines = []
        lines.append("=" * 60)
        lines.append("  HDP-DS 闭环反馈报告")
        lines.append("=" * 60)

        for result in self.cycle_history:
            lines.append(f"\n  周期 {result.cycle}:")
            lines.append(f"    预测 KPI 均值: {result.predictions.mean():.4f}")
            if result.real_kpi is not None:
                lines.append(f"    真实 KPI 均值: {result.real_kpi.mean():.4f}")
                error = np.abs(result.predictions - result.real_kpi).mean()
                lines.append(f"    预测误差 (MAE): {error:.4f}")
            lines.append(f"    反馈应用: {'✓' if result.feedback_applied else '✗'}")

            if result.rankings:
                top3 = result.rankings[:3]
                lines.append(f"    前3推荐:")
                for r in top3:
                    lines.append(f"      #{r.rank} {r.supplier_name:20s} "
                                 f"等级={r.recommendation_level}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def print_report(self):
        """打印闭环报告"""
        print(self.generate_report())

    def get_kpi_trajectory(self) -> np.ndarray:
        """获取 KPI 变化轨迹 [n_cycles, 5]"""
        if not self.kpi_history:
            return np.array([])
        return np.array([h.mean(axis=0) for h in self.kpi_history])
