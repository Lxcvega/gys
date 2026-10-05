"""HDP-DS v3.1 TOPSIS 决策层 — 对 Pareto 解集进行 TOPSIS 排序并输出推荐等级"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TOPSISConfig:
    """TOPSIS 配置"""
    objective_directions: Dict[str, bool] = field(default_factory=lambda: {
        'F1_performance': True, 'F2_risk': False, 'F3_cost': False, 'F4_stability': True
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'strongly_recommended': 0.70, 'recommended': 0.55, 'conditional': 0.40, 'eliminated': 0.00
    })
    level_names: Dict[str, str] = field(default_factory=lambda: {
        'strongly_recommended': 'Strongly Recommended (强烈推荐)',
        'recommended': 'Recommended (推荐)',
        'conditional': 'Conditional (条件推荐)',
        'eliminated': 'Eliminated (淘汰)'
    })


@dataclass
class SupplierRanking:
    """供应商排序结果"""
    supplier_id: str
    supplier_name: str
    closeness: float               # TOPSIS 贴近度
    rank: int                       # 排序
    recommendation_level: str       # 推荐等级
    f1_score: float                 # 综合绩效
    f2_risk: float                  # 风险
    f3_cost: float                  # 成本波动
    f4_stability: float             # 供应稳定性


class TOPSISDecision:
    """TOPSIS 多准则决策: Pareto 解集 → 排序 → 推荐等级"""

    def __init__(self, config: Optional[TOPSISConfig] = None):
        self.config = config or TOPSISConfig()

    def rank(
        self,
        objective_matrix: np.ndarray,   # [n_candidates, 4] Pareto 目标值
        supplier_ids: List[str],
        supplier_names: Optional[List[str]] = None,
        kpi_matrix: Optional[np.ndarray] = None   # [n_candidates, 5] 原始 KPI
    ) -> List[SupplierRanking]:
        """
        TOPSIS 排序

        注意: objective_matrix 是 pymoo 输出 (全部最小化)
        需要转换方向: F1和F4原始为负，需取反恢复
        """
        n_candidates = objective_matrix.shape[0]
        if supplier_names is None:
            supplier_names = [f"供应商{sid}" for sid in supplier_ids]

        # 1. 转换目标值方向 (pymoo 全部最小化，恢复原始方向)
        # F1 (绩效) 和 F4 (稳定性) 取负恢复
        transformed = objective_matrix.copy()
        transformed[:, 0] = -transformed[:, 0]   # F1: 越大越好
        transformed[:, 3] = -transformed[:, 3]   # F4: 越大越好
        # F2 (风险) 和 F3 (成本) 保持越小越好

        # 2. 归一化 (向量归一化)
        norm_matrix = np.zeros_like(transformed)
        for j in range(4):
            col = transformed[:, j]
            norm = np.sqrt(np.sum(col ** 2))
            if norm > 0:
                norm_matrix[:, j] = col / norm
            else:
                norm_matrix[:, j] = col

        # 3. 确定正理想解和负理想解
        ideal_best = np.zeros(4)
        ideal_worst = np.zeros(4)

        # 目标方向: [True, False, False, True]
        directions = [True, False, False, True]  # F1↑ F2↓ F3↓ F4↑

        for j in range(4):
            if directions[j]:  # 越大越好
                ideal_best[j] = norm_matrix[:, j].max()
                ideal_worst[j] = norm_matrix[:, j].min()
            else:              # 越小越好
                ideal_best[j] = norm_matrix[:, j].min()
                ideal_worst[j] = norm_matrix[:, j].max()

        # 4. 计算到正负理想解的距离
        dist_best = np.sqrt(np.sum((norm_matrix - ideal_best) ** 2, axis=1))
        dist_worst = np.sqrt(np.sum((norm_matrix - ideal_worst) ** 2, axis=1))

        # 5. 计算相对贴近度
        denominator = dist_best + dist_worst
        denominator[denominator == 0] = 1e-10
        closeness = dist_worst / denominator

        # 6. 排序
        sorted_indices = np.argsort(-closeness)

        rankings = []
        for rank, idx in enumerate(sorted_indices):
            c = closeness[idx]
            level = self._determine_level(c)

            ranking = SupplierRanking(
                supplier_id=supplier_ids[idx],
                supplier_name=supplier_names[idx],
                closeness=c,
                rank=rank + 1,
                recommendation_level=level,
                f1_score=float(transformed[idx, 0]),
                f2_risk=float(transformed[idx, 1]),
                f3_cost=float(transformed[idx, 2]),
                f4_stability=float(transformed[idx, 3])
            )
            rankings.append(ranking)

        return rankings

    def rank_suppliers_from_kpi(
        self,
        kpi_matrix: np.ndarray,          # [n_suppliers, 5]
        supplier_ids: List[str],
        supplier_names: Optional[List[str]] = None,
        kpi_weights: Optional[List[float]] = None
    ) -> List[SupplierRanking]:
        """
        直接从 KPI 矩阵对供应商进行 TOPSIS 排序

        用于 NSGA-II 不可用或需要快速排序的场景。
        """
        if kpi_weights is None:
            kpi_weights = [0.25, 0.25, 0.15, 0.15, 0.20]

        kw = np.array(kpi_weights)

        # 计算 4 个目标值
        n = kpi_matrix.shape[0]
        objectives = np.zeros((n, 4))
        for i in range(n):
            p = kpi_matrix[i]
            objectives[i, 0] = kw[0] * p[0] + kw[1] * p[1] + kw[2] * (1 - p[2]) + kw[3] * p[3] + kw[4] * p[4]
            objectives[i, 1] = p[2]  # 环境风险
            objectives[i, 2] = 1.0 - p[3]  # 成本波动
            objectives[i, 3] = (p[0] + p[1]) / 2.0  # 供应稳定性

        return self.rank(objectives, supplier_ids, supplier_names, kpi_matrix)

    def determine_levels(
        self, rankings: List[SupplierRanking]
    ) -> Dict[str, List[SupplierRanking]]:
        """按推荐等级分组"""
        levels = {name: [] for name in self.config.level_names.values()}
        for r in rankings:
            levels[r.recommendation_level].append(r)
        return levels

    def print_ranking_report(self, rankings: List[SupplierRanking]):
        """打印排序报告"""
        levels = self.determine_levels(rankings)

        print("\n" + "=" * 65)
        print("  TOPSIS 供应商排序报告")
        print("=" * 65)

        for level_name, level_rankings in levels.items():
            if not level_rankings:
                continue
            # 确定等级标记
            if 'Strongly' in level_name:
                marker = "★★★"
            elif 'Recomm' in level_name and 'Strongly' not in level_name:
                marker = "★★"
            elif 'Conditional' in level_name:
                marker = "★"
            else:
                marker = "[E]"

            print(f"\n  {marker} {level_name}:")
            print(f"  {'─' * 60}")
            for r in level_rankings:
                print(f"    #{r.rank:2d} | {r.supplier_name:20s} | "
                      f"贴近度={r.closeness:.4f} | "
                      f"绩效={r.f1_score:.3f} 风险={r.f2_risk:.3f} "
                      f"成本={r.f3_cost:.3f} 稳定={r.f4_stability:.3f}")

        print(f"\n  {'=' * 65}")

    def _determine_level(self, closeness: float) -> str:
        """根据贴近度确定推荐等级"""
        if closeness >= self.config.thresholds['strongly_recommended']:
            return self.config.level_names['strongly_recommended']
        elif closeness >= self.config.thresholds['recommended']:
            return self.config.level_names['recommended']
        elif closeness >= self.config.thresholds['conditional']:
            return self.config.level_names['conditional']
        else:
            return self.config.level_names['eliminated']
