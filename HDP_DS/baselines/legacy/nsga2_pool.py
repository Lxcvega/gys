import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .data_model import (
    SupplierCapability, WastewaterType,
    SupplierPoolMember, ChemicalRequirement, SupplierProposal
)
@dataclass
class OptimizationObjective:
    name: str
    weight: float
    minimize: bool
    target_value: Optional[float] = None

@dataclass
class ParetoFront:
    front_number: int
    solutions: List[SupplierPoolMember]
    crowding_distances: List[float]

class NSGA2SupplierPoolBuilder:
    def __init__(self):
        self.population_size = 0
        self.max_generations = 20
        self.crossover_rate = 0.9
        # 默认使用自动调参发现的更优初始参数
        self.mutation_rate = 0.02
        self.tournament_size = 3
        # 默认的种群规模乘数（在没有显式设置 population_size 时使用）
        self.default_population_multiplier = 1.5
        
        self.objectives = [
            OptimizationObjective('cost', 0.3, True),
            OptimizationObjective('quality', 0.25, False),
            OptimizationObjective('delivery', 0.2, True),
            OptimizationObjective('environmental', 0.15, False),
            OptimizationObjective('reliability', 0.1, False)
        ]
        
        self.supplier_pool: List[SupplierPoolMember] = []
        self.pareto_fronts: List[ParetoFront] = []
        
        self.wastewater_type_weights = {
            WastewaterType.INDUSTRIAL: {'cost': 0.25, 'quality': 0.30, 'delivery': 0.15, 'environmental': 0.20, 'reliability': 0.10},
            WastewaterType.DOMESTIC: {'cost': 0.35, 'quality': 0.20, 'delivery': 0.20, 'environmental': 0.15, 'reliability': 0.10},
            WastewaterType.MEDICAL: {'cost': 0.20, 'quality': 0.35, 'delivery': 0.15, 'environmental': 0.20, 'reliability': 0.10},
            WastewaterType.AGRICULTURAL: {'cost': 0.30, 'quality': 0.20, 'delivery': 0.25, 'environmental': 0.15, 'reliability': 0.10},
            WastewaterType.CHEMICAL: {'cost': 0.25, 'quality': 0.25, 'delivery': 0.15, 'environmental': 0.25, 'reliability': 0.10}
        }

    def set_objectives_for_wastewater_type(self, wastewater_type: WastewaterType):
        type_weights = self.wastewater_type_weights.get(wastewater_type, {})
        for objective in self.objectives:
            if objective.name in type_weights:
                objective.weight = type_weights[objective.name]

    def calculate_objectives(self, supplier: SupplierCapability, 
                            proposal: SupplierProposal,
                            requirement: ChemicalRequirement,
                            nn_reliability_score: Optional[float] = None) -> Dict[str, float]:
        """计算五目标值
        修复: f5(Reliability) 不再使用人工加权公式，而是优先使用
        BP神经网络从历史数据中学习到的可靠性预测值 nn_reliability_score。
        只有当BP模型未训练或无预测时，才回退到人工加权兜底。
        """
        objectives = {}
        
        objectives['cost'] = min(1.0, proposal.unit_price * proposal.offered_quantity / requirement.budget_constraint)
        
        objectives['quality'] = (
            (supplier.effective_ingredient_content / 100.0) * 0.4 +
            (supplier.delivery_timeliness / 100.0) * 0.3 +
            (1.0 if len(proposal.quality_certification) > 0 else 0.0) * 0.3
        )
        
        objectives['delivery'] = proposal.delivery_time / 30.0
        
        objectives['environmental'] = (
            supplier.safety_management_level / 5.0 * 0.5 +
            (1.0 if len(proposal.environmental_certification) > 0 else 0.0) * 0.3 +
            supplier.inventory_capacity / 10000.0 * 0.2
        )
        
        # ★ 修复: f5 使用BP神经网络的数据驱动预测，替代人工加权公式
        if nn_reliability_score is not None:
            objectives['reliability'] = nn_reliability_score
        else:
            # 兜底方案：人工加权（仅在BP模型不可用时使用）
            objectives['reliability'] = (
                supplier.delivery_capability / 5.0 * 0.4 +
                supplier.emergency_supply_capability / 5.0 * 0.3 +
                supplier.safety_management_level / 5.0 * 0.3
            )
        
        return objectives

    def dominates(self, solution1: SupplierPoolMember, solution2: SupplierPoolMember) -> bool:
        obj1 = solution1.objectives
        obj2 = solution2.objectives
        
        # 防御：检查 objectives 类型
        if not isinstance(obj1, dict):
            raise TypeError(f"[NSGA2] solution1.objectives 类型错误: {type(obj1).__name__} (值={obj1}), supplier_id={solution1.supplier_id}")
        if not isinstance(obj2, dict):
            raise TypeError(f"[NSGA2] solution2.objectives 类型错误: {type(obj2).__name__} (值={obj2}), supplier_id={solution2.supplier_id}")
        
        at_least_one_better = False
        all_better_or_equal = True
        
        for objective in self.objectives:
            val1 = obj1.get(objective.name, 0)
            val2 = obj2.get(objective.name, 0)
            
            if objective.minimize:
                if val1 < val2:
                    at_least_one_better = True
                elif val1 > val2:
                    all_better_or_equal = False
            else:
                if val1 > val2:
                    at_least_one_better = True
                elif val1 < val2:
                    all_better_or_equal = False
        
        return at_least_one_better and all_better_or_equal

    def fast_non_dominated_sort(self, population: List[SupplierPoolMember]) -> List[ParetoFront]:
        n = len(population)
        if n == 0:
            return []
        
        fronts = []
        
        # 初始化支配结构
        for solution in population:
            solution.dominance_count = 0
            solution.dominated_solutions = []
        
        # 计算支配关系（使用对象引用，避免依赖 supplier_id 唯一性）
        for i, solution in enumerate(population):
            for j, other in enumerate(population):
                if i == j:
                    continue
                if self.dominates(solution, other):
                    solution.dominated_solutions.append(other)
                elif self.dominates(other, solution):
                    solution.dominance_count += 1
        
        # 第一前沿
        current_front = [s for s in population if s.dominance_count == 0]
        for s in current_front:
            s.pareto_rank = 0
        fronts.append(ParetoFront(front_number=0, solutions=current_front, crowding_distances=[]))
        
        i = 0
        while i < len(fronts) and len(fronts[i].solutions) > 0:
            next_front = []
            for solution in fronts[i].solutions:
                for dominated_solution in solution.dominated_solutions:
                    dominated_solution.dominance_count -= 1
                    if dominated_solution.dominance_count == 0:
                        dominated_solution.pareto_rank = i + 1
                        next_front.append(dominated_solution)
            i += 1
            if next_front:
                fronts.append(ParetoFront(front_number=i, solutions=next_front, crowding_distances=[]))
            else:
                break
        
        return fronts

    def calculate_crowding_distance(self, front: ParetoFront) -> List[float]:
        solutions = front.solutions
        num_solutions = len(solutions)
        
        if num_solutions == 0:
            return []
        
        if num_solutions <= 2:
            return [float('inf')] * num_solutions
        
        distances = [0.0] * num_solutions
        
        # 对每个目标进行规范化并累加距离（采用常见实现，最大化/最小化统一处理）
        for objective in self.objectives:
            obj_name = objective.name
            # 提取目标值列表
            values = [s.objectives.get(obj_name, 0.0) if isinstance(s.objectives, dict) else 0.0 for s in solutions]
            # 排序索引
            sorted_indices = sorted(range(num_solutions), key=lambda i: values[i])
            min_obj = values[sorted_indices[0]]
            max_obj = values[sorted_indices[-1]]
            obj_range = max_obj - min_obj
            
            # 若范围为0，则该目标对拥挤度无贡献
            if obj_range == 0:
                # 将边界点设置为无限拥挤距离
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                continue
            
            # 边界点
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            for idx in range(1, num_solutions - 1):
                i = sorted_indices[idx]
                prev_val = values[sorted_indices[idx - 1]]
                next_val = values[sorted_indices[idx + 1]]
                # 统一使用差值绝对值并归一化
                distance = abs(next_val - prev_val) / obj_range
                distances[i] += distance
        
        return distances

    def tournament_selection(self, population: List[SupplierPoolMember]) -> SupplierPoolMember:
        candidates = random.sample(population, min(self.tournament_size, len(population)))
        return min(candidates, key=lambda s: (s.pareto_rank, -s.crowding_distance))

    def crossover(self, parent1: SupplierPoolMember, parent2: SupplierPoolMember) -> Tuple[SupplierPoolMember, SupplierPoolMember]:
        # 交叉操作：子代继承父代的supplier_id，确保可追溯回真实供应商
        child1 = SupplierPoolMember(
            supplier_id=parent1.supplier_id,
            pareto_rank=0,
            crowding_distance=0.0,
            objectives={},
            dominance_count=0,
            dominated_solutions=[]
        )
        
        child2 = SupplierPoolMember(
            supplier_id=parent2.supplier_id,
            pareto_rank=0,
            crowding_distance=0.0,
            objectives={},
            dominance_count=0,
            dominated_solutions=[]
        )
        
        for objective in self.objectives:
            obj_name = objective.name
            val1 = parent1.objectives.get(obj_name, 0)
            val2 = parent2.objectives.get(obj_name, 0)
            
            alpha = random.random()
            child1.objectives[obj_name] = alpha * val1 + (1 - alpha) * val2
            child2.objectives[obj_name] = (1 - alpha) * val1 + alpha * val2
        
        return child1, child2

    def mutate(self, solution: SupplierPoolMember) -> SupplierPoolMember:
        # 变异操作：保持supplier_id不变
        mutated = SupplierPoolMember(
            supplier_id=solution.supplier_id,
            pareto_rank=solution.pareto_rank,
            crowding_distance=solution.crowding_distance,
            objectives=solution.objectives.copy(),
            dominance_count=solution.dominance_count,
            dominated_solutions=solution.dominated_solutions.copy()
        )
        
        for objective in self.objectives:
            if random.random() < self.mutation_rate:
                obj_name = objective.name
                mutation = random.gauss(0, 0.1)
                mutated.objectives[obj_name] = max(0, min(1, mutated.objectives.get(obj_name, 0) + mutation))
        
        return mutated

    def create_initial_population(self, suppliers: List[SupplierCapability],
                                  proposals: Dict[str, SupplierProposal],
                                  requirement: ChemicalRequirement,
                                  nn_reliability_scores: Optional[Dict[str, float]] = None
                                  ) -> List[SupplierPoolMember]:
        population = []
        
        for supplier in suppliers:
            proposal = proposals.get(supplier.supplier_id)
            if not proposal:
                continue
            
            # 获取BP神经网络对该供应商的可靠性预测（若有）
            nn_score = None
            if nn_reliability_scores is not None:
                nn_score = nn_reliability_scores.get(supplier.supplier_id)
            
            objectives = self.calculate_objectives(supplier, proposal, requirement, nn_score)
            
            member = SupplierPoolMember(
                supplier_id=supplier.supplier_id,
                pareto_rank=0,
                crowding_distance=0.0,
                objectives=objectives,
                dominance_count=0,
                dominated_solutions=[]
            )
            population.append(member)
        
        # 不再填充合成个体——初始种群仅包含真实供应商
        return population

    def evolve_population(self, population: List[SupplierPoolMember]) -> List[SupplierPoolMember]:
        fronts = self.fast_non_dominated_sort(population)
        
        for front in fronts:
            distances = self.calculate_crowding_distance(front)
            front.crowding_distances = distances
            for i, solution in enumerate(front.solutions):
                solution.crowding_distance = distances[i]
        
        new_population = []
        i = 0
        while i < len(fronts) and len(new_population) + len(fronts[i].solutions) <= self.population_size:
            new_population.extend(fronts[i].solutions)
            i += 1
        
        if i < len(fronts):
            remaining = self.population_size - len(new_population)
            sorted_solutions = sorted(
                fronts[i].solutions,
                key=lambda s: s.crowding_distance,
                reverse=True
            )
            new_population.extend(sorted_solutions[:remaining])
        
        offspring = []
        offspring_needed = self.population_size
        while len(offspring) < offspring_needed:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            offspring.extend([child1, child2])
        
        combined_population = new_population + offspring[:self.population_size]
        
        return combined_population

    def build_supplier_pool(self, suppliers: List[SupplierCapability],
                           proposals: Dict[str, SupplierProposal],
                           requirement: ChemicalRequirement,
                           wastewater_type: WastewaterType,
                           nn_reliability_scores: Optional[Dict[str, float]] = None
                           ) -> List[SupplierPoolMember]:
        """构建NSGA-II供应商池
        
        Args:
            nn_reliability_scores: BP神经网络预测的供应商可靠性得分 {supplier_id: score}。
                传入后取代 f5(Reliability) 的人工加权公式，实现数据驱动的多目标优化。
        """
        self.set_objectives_for_wastewater_type(wastewater_type)
        
        # 种群大小：根据默认乘数或显式设置决定
        num_suppliers = len(suppliers)
        if getattr(self, 'population_size', 0) <= 0:
            # 使用默认乘数扩充种群，以便优化多样性
            self.population_size = max(num_suppliers, int(num_suppliers * getattr(self, 'default_population_multiplier', 1.5)))
        elif self.population_size < num_suppliers:
            self.population_size = num_suppliers

        population = self.create_initial_population(suppliers, proposals, requirement, nn_reliability_scores)
        
        # 保存初始种群（所有供应商的信息），供后续补充被淘汰的供应商使用
        initial_population = {m.supplier_id: m for m in population}
        
        # 运行NSGA-II进化（进化后部分供应商可能被淘汰）
        for generation in range(self.max_generations):
            population = self.evolve_population(population)
            # 保存每代 Pareto 前沿以便后续计算收敛性指标
            try:
                fronts_gen = self.fast_non_dominated_sort(population)
                if not hasattr(self, 'pareto_history'):
                    self.pareto_history = []
                first_front_solutions = [
                    {obj.name: s.objectives.get(obj.name, 0.0) for obj in self.objectives}
                    for s in (fronts_gen[0].solutions if fronts_gen else [])
                    if isinstance(s.objectives, dict)
                ]
                if first_front_solutions:
                    self.pareto_history.append(first_front_solutions)
            except Exception:
                pass
        
        # 对进化后的种群做非支配排序
        self.pareto_fronts = self.fast_non_dominated_sort(population)
        
        for front in self.pareto_fronts:
            distances = self.calculate_crowding_distance(front)
            front.crowding_distances = distances
            for i, solution in enumerate(front.solutions):
                solution.crowding_distance = distances[i]
        
        # ★ 修复：收集进化后每个供应商ID的最佳（最低）Pareto rank
        best_rank_per_supplier: Dict[str, int] = {}
        best_member_per_supplier: Dict[str, SupplierPoolMember] = {}
        for member in population:
            sid = member.supplier_id
            if sid not in best_rank_per_supplier or member.pareto_rank < best_rank_per_supplier[sid]:
                best_rank_per_supplier[sid] = member.pareto_rank
                best_member_per_supplier[sid] = member
        
        # ★ 为被淘汰的供应商分配最差rank+1，确保所有供应商都有NSGA-II排序记录
        max_rank = max(best_rank_per_supplier.values()) if best_rank_per_supplier else 0
        for sid, init_member in initial_population.items():
            if sid not in best_member_per_supplier:
                init_member.pareto_rank = max_rank + 1
                init_member.crowding_distance = 0.0
                best_member_per_supplier[sid] = init_member
        
        # 返回包含全部供应商的去重池
        all_ranked = list(best_member_per_supplier.values())
        self.supplier_pool = all_ranked
        
        return self.supplier_pool

    def get_pareto_optimal_suppliers(self, top_n: Optional[int] = None) -> List[SupplierPoolMember]:
        if not self.pareto_fronts:
            return []
        
        # 如果没有指定top_n，则根据供应商池大小动态设置
        if top_n is None:
            pool_size = len(self.supplier_pool)
            top_n = min(pool_size, max(10, pool_size // 2))
        
        pareto_optimal = []
        remaining = top_n
        
        for front in self.pareto_fronts:
            if remaining <= 0:
                break
            
            sorted_solutions = sorted(
                front.solutions,
                key=lambda s: s.crowding_distance,
                reverse=True
            )
            
            take = min(remaining, len(sorted_solutions))
            pareto_optimal.extend(sorted_solutions[:take])
            remaining -= take
        
        return pareto_optimal

    def calculate_hypervolume(self, reference_point: Dict[str, float]) -> float:
        if not self.pareto_fronts:
            return 0.0
        
        pareto_front = self.pareto_fronts[0].solutions
        if not pareto_front:
            return 0.0
        
        hypervolume = 0.0
        
        for solution in pareto_front:
            volume = 1.0
            for objective in self.objectives:
                obj_name = objective.name
                value = solution.objectives.get(obj_name, 0)
                ref_value = reference_point.get(obj_name, 1.0)
                
                if objective.minimize:
                    contribution = max(0, ref_value - value)
                else:
                    contribution = max(0, value)
                
                volume *= contribution
            
            hypervolume += volume
        
        return hypervolume

    def generate_pool_report(self) -> str:
        report = "NSGA-II 供应商池构建报告\n"
        report += "=" * 60 + "\n\n"
        
        report += f"优化目标:\n"
        for obj in self.objectives:
            direction = "最小化" if obj.minimize else "最大化"
            report += f"• {obj.name}: 权重={obj.weight:.2f}, {direction}\n"
        
        report += f"\nPareto前沿面数量: {len(self.pareto_fronts)}\n"
        report += f"供应商池大小: {len(self.supplier_pool)}\n\n"
        
        if self.pareto_fronts:
            report += "第一前沿面 (Pareto最优解):\n"
            report += "-" * 60 + "\n"
            
            sorted_front = sorted(
                self.pareto_fronts[0].solutions,
                key=lambda s: s.crowding_distance,
                reverse=True
            )
            
            for i, solution in enumerate(sorted_front[:10], 1):
                report += f"{i}. 供应商ID: {solution.supplier_id}\n"
                report += f"   拥挤距离: {solution.crowding_distance:.4f}\n"
                report += "   目标值:\n"
                for obj in self.objectives:
                    value = solution.objectives.get(obj.name, 0)
                    report += f"     • {obj.name}: {value:.4f}\n"
                report += "\n"
        
        return report