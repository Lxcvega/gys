"""HDP-DS v3.1 NSGA-II 多目标优化模块 — 基于 pymoo 实现 4 目标优化"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("  [NSGA-II] 警告: pymoo 未安装，将使用简化版 NSGA-II")


@dataclass
class NSGA2Config:
    """NSGA-II 配置"""
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'F1_performance': 0.30, 'F2_risk': 0.25, 'F3_cost': 0.20, 'F4_stability': 0.25
    })
    pop_size: int = 100
    n_offsprings: int = 50
    n_gen: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    seed: int = 42
    verbose: bool = False
    kpi_weights: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.15, 0.15, 0.20])


# ─── pymoo 问题定义 ─────────────────────────────────────────────

class HDPSupplierProblem(ElementwiseProblem):
    """
    HDP-DS 供应商选择问题 (4 目标)

    决策变量: 供应商选择权重 [0, 1]
      每个变量代表一个供应商被选中的倾向

    输入: KPI 矩阵 P (每个供应商 5 维 KPI 预测)
    """

    def __init__(self, kpi_matrix: np.ndarray, kpi_weights: List[float]):
        """
        Args:
            kpi_matrix: [n_suppliers, 5] KPI 预测矩阵
            kpi_weights: 5 维 KPI 权重 (用于 F1)
        """
        self.kpi_matrix = kpi_matrix
        self.kpi_weights = np.array(kpi_weights)
        self.n_suppliers = kpi_matrix.shape[0]

        # 4 目标, n_suppliers 个决策变量
        super().__init__(
            n_var=self.n_suppliers,
            n_obj=4,
            xl=0.0,
            xu=1.0
        )

    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """
        评估 4 个目标

        x: [n_suppliers] 供应商选择权重向量
        """
        # 归一化权重
        w = x / (x.sum() + 1e-10)

        # 加权 KPI 向量
        weighted_kpi = w @ self.kpi_matrix  # [5]

        p1, p2, p3, p4, p5 = weighted_kpi

        # 目标1: 最大化综合绩效 (取负)
        f1 = -(self.kpi_weights[0] * p1 +
               self.kpi_weights[1] * p2 +
               self.kpi_weights[2] * (1 - p3) +
               self.kpi_weights[3] * p4 +
               self.kpi_weights[4] * p5)

        # 目标2: 最小化风险 (= p3 环境风险)
        f2 = p3

        # 目标3: 最小化成本波动 (= 1 - p4 成本稳定性逆)
        f3 = 1.0 - p4

        # 目标4: 最大化供应稳定性 (取负)
        f4 = -(p1 + p2) / 2.0

        out["F"] = np.array([f1, f2, f3, f4], dtype=float)


# ─── 简化版 NSGA-II (当 pymoo 不可用时) ───────────────────────

class SimpleNSGA2:
    """
    简化版 NSGA-II (纯 NumPy 实现)

    仅当 pymoo 不可用时作为 fallback。
    """

    def __init__(self, config: NSGA2Config):
        self.config = config

    def optimize(self, kpi_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        简化 NSGA-II 优化

        Returns:
            (pareto_solutions, pareto_front)
        """
        n_suppliers = kpi_matrix.shape[0]
        pop_size = min(self.config.pop_size, 500)

        # 随机初始化种群
        population = np.random.rand(pop_size, n_suppliers)
        fitness = np.array([self._evaluate(ind, kpi_matrix)
                           for ind in population])

        for gen in range(self.config.n_gen):
            # 非支配排序
            fronts = self._fast_non_dominated_sort(fitness)

            # 锦标赛选择
            offspring = []
            for _ in range(pop_size // 2):
                p1 = self._tournament_selection(population, fitness, fronts)
                p2 = self._tournament_selection(population, fitness, fronts)

                # 交叉
                alpha = np.random.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = (1 - alpha) * p1 + alpha * p2

                # 变异
                if np.random.rand() < self.config.mutation_prob:
                    c1 += np.random.randn(n_suppliers) * 0.05
                    c2 += np.random.randn(n_suppliers) * 0.05

                c1 = np.clip(c1, 0, 1)
                c2 = np.clip(c2, 0, 1)
                offspring.extend([c1, c2])

            offspring = np.array(offspring[:pop_size])
            offspring_fitness = np.array([self._evaluate(ind, kpi_matrix)
                                          for ind in offspring])

            # 合并种群
            combined_pop = np.vstack([population, offspring])
            combined_fit = np.vstack([fitness, offspring_fitness])

            # 环境选择 (NSGA-II 拥挤度)
            selected = self._environmental_selection(combined_pop, combined_fit, pop_size)
            population = combined_pop[selected]
            fitness = combined_fit[selected]

        # 获取 Pareto 前沿
        fronts = self._fast_non_dominated_sort(fitness)
        pareto_indices = fronts[0]
        pareto_solutions = population[pareto_indices]
        pareto_front = fitness[pareto_indices]

        return pareto_solutions, pareto_front

    def _evaluate(self, x: np.ndarray, kpi_matrix: np.ndarray) -> np.ndarray:
        w = x / (x.sum() + 1e-10)
        weighted_kpi = w @ kpi_matrix
        p1, p2, p3, p4, p5 = weighted_kpi
        kw = self.config.kpi_weights

        f1 = -(kw[0] * p1 + kw[1] * p2 + kw[2] * (1 - p3) + kw[3] * p4 + kw[4] * p5)
        f2 = p3
        f3 = 1.0 - p4
        f4 = -(p1 + p2) / 2.0
        return np.array([f1, f2, f3, f4])

    def _fast_non_dominated_sort(self, fitness: np.ndarray) -> List[np.ndarray]:
        n = fitness.shape[0]
        domination_count = np.zeros(n)
        dominated_set = [[] for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(fitness[i], fitness[j]):
                    dominated_set[i].append(j)
                elif self._dominates(fitness[j], fitness[i]):
                    domination_count[i] += 1

        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(np.array(current_front))
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _dominates(self, f1: np.ndarray, f2: np.ndarray) -> bool:
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _tournament_selection(self, pop, fitness, fronts, k=3):
        n = len(pop)
        candidates = np.random.choice(n, k, replace=False)
        # 按 Pareto rank 升序，拥挤度降序
        ranks = np.array([self._get_rank(i, fronts) for i in candidates])
        best = candidates[np.argmin(ranks)]
        return pop[best]

    def _get_rank(self, idx, fronts):
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)

    def _environmental_selection(self, pop, fitness, n_select):
        fronts = self._fast_non_dominated_sort(fitness)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(front.tolist())
            else:
                # 拥挤度排序
                front_fitness = fitness[front]
                crowding = self._crowding_distance(front_fitness)
                sorted_idx = np.argsort(-crowding)
                remaining = n_select - len(selected)
                selected.extend(front[sorted_idx[:remaining]].tolist())
                break
        return np.array(selected)

    def _crowding_distance(self, fitness):
        n, m = fitness.shape
        if n <= 2:
            return np.full(n, np.inf)
        distances = np.zeros(n)
        for j in range(m):
            idx = np.argsort(fitness[:, j])
            distances[idx[0]] = np.inf
            distances[idx[-1]] = np.inf
            norm = fitness[idx[-1], j] - fitness[idx[0], j]
            if norm > 0:
                for k in range(1, n - 1):
                    distances[idx[k]] += (fitness[idx[k + 1], j] - fitness[idx[k - 1], j]) / norm
        return distances


# ─── HDP-NSGA-II 封装 ───────────────────────────────────────────

class HDPNSGA2:
    """
    HDP-DS NSGA-II 封装

    输入: 未来 KPI 预测向量 P [n_suppliers, 5]
    输出: Pareto Front 解集 + 每个解的目标值
    """

    def __init__(self, config: Optional[NSGA2Config] = None):
        self.config = config or NSGA2Config()
        self.pareto_solutions: Optional[np.ndarray] = None
        self.pareto_front: Optional[np.ndarray] = None
        self.supplier_ids: List[str] = []

    def optimize(
        self,
        kpi_matrix: np.ndarray,
        supplier_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行 NSGA-II 优化

        Args:
            kpi_matrix: [n_suppliers, 5] KPI 预测矩阵
            supplier_ids: 供应商 ID 列表

        Returns:
            (pareto_solutions, pareto_front)
            pareto_solutions: [n_pareto, n_suppliers] 权重向量
            pareto_front:     [n_pareto, 4] 目标值
        """
        n_suppliers = kpi_matrix.shape[0]
        if supplier_ids is not None:
            self.supplier_ids = supplier_ids
        else:
            self.supplier_ids = [f"S{i}" for i in range(n_suppliers)]

        if n_suppliers < 3:
            print(f"  [NSGA-II] 供应商数量不足 ({n_suppliers} < 3)，跳过优化")
            return np.array([]), np.array([])

        if PYMOO_AVAILABLE:
            # 使用 pymoo
            problem = HDPSupplierProblem(kpi_matrix, self.config.kpi_weights)
            algorithm = PymooNSGA2(
                pop_size=self.config.pop_size,
                n_offsprings=self.config.n_offsprings,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=self.config.crossover_prob, eta=15),
                mutation=PM(prob=self.config.mutation_prob, eta=20),
                eliminate_duplicates=True
            )
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.config.n_gen),
                seed=self.config.seed,
                verbose=self.config.verbose
            )
            self.pareto_solutions = res.X
            self.pareto_front = res.F
        else:
            # 使用简化版
            print("  [NSGA-II] 使用简化版 (pymoo 未安装)")
            solver = SimpleNSGA2(self.config)
            self.pareto_solutions, self.pareto_front = solver.optimize(kpi_matrix)

        if self.config.verbose:
            print(f"  [NSGA-II] Pareto 解数: {len(self.pareto_solutions)}")

        return self.pareto_solutions, self.pareto_front

    def get_pareto_summary(self) -> Dict[str, Any]:
        """获取 Pareto 前沿摘要"""
        if self.pareto_front is None or len(self.pareto_front) == 0:
            return {'n_solutions': 0}

        pf = self.pareto_front
        summary = {
            'n_solutions': len(pf),
            'F1_performance': {
                'min': float(-pf[:, 0].max()),
                'max': float(-pf[:, 0].min()),
                'mean': float(-pf[:, 0].mean()),
            },
            'F2_risk': {
                'min': float(pf[:, 1].min()),
                'max': float(pf[:, 1].max()),
                'mean': float(pf[:, 1].mean()),
            },
            'F3_cost': {
                'min': float(pf[:, 2].min()),
                'max': float(pf[:, 2].max()),
                'mean': float(pf[:, 2].mean()),
            },
            'F4_stability': {
                'min': float(-pf[:, 3].max()),
                'max': float(-pf[:, 3].min()),
                'mean': float(-pf[:, 3].mean()),
            },
        }
        return summary

    def print_summary(self):
        """打印 Pareto 前沿摘要"""
        summary = self.get_pareto_summary()
        if summary['n_solutions'] == 0:
            print("  [NSGA-II] 无 Pareto 解")
            return

        print(f"\n  {'=' * 50}")
        print(f"  NSGA-II Pareto 前沿摘要")
        print(f"  {'=' * 50}")
        print(f"  Pareto 解数: {summary['n_solutions']}")
        print(f"  F1 综合绩效: {summary['F1_performance']['mean']:.4f} "
              f"[{summary['F1_performance']['min']:.4f}, {summary['F1_performance']['max']:.4f}]")
        print(f"  F2 风险:     {summary['F2_risk']['mean']:.4f} "
              f"[{summary['F2_risk']['min']:.4f}, {summary['F2_risk']['max']:.4f}]")
        print(f"  F3 成本波动: {summary['F3_cost']['mean']:.4f} "
              f"[{summary['F3_cost']['min']:.4f}, {summary['F3_cost']['max']:.4f}]")
        print(f"  F4 稳定性:   {summary['F4_stability']['mean']:.4f} "
              f"[{summary['F4_stability']['min']:.4f}, {summary['F4_stability']['max']:.4f}]")
        print(f"  {'=' * 50}")
