from typing import Dict, List, Tuple
from data_model import SupplierScoringData, SupplierScoringResult

SCORING_RANGES = {
    'material_unit_price':   (800,  5000),
    'transport_cost':        (177,  401),
    'payment_cycle':         (7,    60),
    'effective_ingredient':  (85,   98),
    'sludge_treatment_rate': (80,   99),
    'delivery_timeliness':   (85,   100),
    'inventory_capacity':    (100,  1000),
    'delivery_capability':   (1,    5),
    'emergency_supply':      (3,    5),
    'safety_management':     (3,    5),
}

class SupplierScoringModule:
    def __init__(self, ranges: Dict[str, Tuple[float, float]] | None = None):
        self._ranges = ranges or SCORING_RANGES
        self.max_scores = {
            'material_unit_price': 18,
            'transport_cost': 9,
            'payment_cycle': 4,
            'effective_ingredient': 15,
            'sludge_treatment_rate': 20,
            'delivery_timeliness': 10,
            'inventory_capacity': 4,
            'delivery_capability': 3,
            'emergency_supply': 5,
            'safety_management': 12
        }
        self.price_max: float = 1.0
        self.price_min: float = 0.0
    
    def calculate_data_ranges(self, scoring_data_list: List[SupplierScoringData]):
        if not scoring_data_list:
            return
        
        material_prices = [data.material_unit_price for data in scoring_data_list]
        self.price_max = max(material_prices)
        self.price_min = min(material_prices)
        
        print("\n=== 评分数据概况（使用可配置行业标准范围） ===")
        print(f"药剂单价范围: {self.price_min:.2f} - {self.price_max:.2f} 元/吨（使用实际最大最小值）")
        print(f"运输成本: 可配置范围 {self._ranges['transport_cost']} 元/吨")
        print(f"付款周期: 可配置范围 {self._ranges['payment_cycle']} 天")
        print(f"药剂有效成分含量: 可配置范围 {self._ranges['effective_ingredient']} %")
        print(f"处理污泥效果达标率: 可配置范围 {self._ranges['sludge_treatment_rate']} %")
        print(f"供货及时性: 可配置范围 {self._ranges['delivery_timeliness']} %")
        print(f"库存能力: 可配置范围 {self._ranges['inventory_capacity']} 吨")
        print(f"交付能力: 可配置范围 {self._ranges['delivery_capability']} 级")
        print(f"应急供应能力: 可配置范围 {self._ranges['emergency_supply']} 级")
        print(f"安全管理水平: 可配置范围 {self._ranges['safety_management']} 级")

    def calculate_material_unit_price_score(self, unit_price: float) -> float:
        if not hasattr(self, 'price_max') or self.price_max == self.price_min:
            return self.max_scores['material_unit_price']
        score = 18 * (self.price_max - unit_price) / (self.price_max - self.price_min)
        return max(0, min(18, score))

    def _linear_score(self, value: float, range_key: str, max_score: float, 
                      higher_is_better: bool) -> float:
        lo, hi = self._ranges.get(range_key, (0, 100))
        span = hi - lo
        if span <= 0:
            return max_score * 0.5
        if higher_is_better:
            score = max_score * (value - lo) / span
        else:
            score = max_score * (hi - value) / span
        return max(0, min(max_score, score))

    def calculate_transport_cost_score(self, transport_cost: float) -> float:
        return self._linear_score(transport_cost, 'transport_cost', 9, higher_is_better=False)

    def calculate_payment_cycle_score(self, payment_cycle: float) -> float:
        return self._linear_score(payment_cycle, 'payment_cycle', 4, higher_is_better=False)

    def calculate_effective_ingredient_score(self, effective_ingredient: float) -> float:
        return self._linear_score(effective_ingredient, 'effective_ingredient', 15, higher_is_better=True)

    def calculate_sludge_treatment_rate_score(self, rate: float) -> float:
        return self._linear_score(rate, 'sludge_treatment_rate', 20, higher_is_better=True)

    def calculate_delivery_timeliness_score(self, timeliness: float) -> float:
        return self._linear_score(timeliness, 'delivery_timeliness', 10, higher_is_better=True)

    def calculate_inventory_capacity_score(self, inventory_capacity: float) -> float:
        return self._linear_score(inventory_capacity, 'inventory_capacity', 4, higher_is_better=True)

    def calculate_delivery_capability_score(self, delivery_capability: int) -> float:
        return self._linear_score(delivery_capability, 'delivery_capability', 3, higher_is_better=True)

    def calculate_emergency_supply_score(self, emergency_level: int) -> float:
        return self._linear_score(emergency_level, 'emergency_supply', 5, higher_is_better=True)

    def calculate_safety_management_score(self, safety_level: int) -> float:
        return self._linear_score(safety_level, 'safety_management', 12, higher_is_better=True)

    def calculate_supplier_score(self, scoring_data: SupplierScoringData, 
                                supplier_name: str = "") -> SupplierScoringResult:
        material_unit_price_score = self.calculate_material_unit_price_score(
            scoring_data.material_unit_price
        )
        transport_cost_score = self.calculate_transport_cost_score(scoring_data.transport_cost)
        payment_cycle_score = self.calculate_payment_cycle_score(scoring_data.payment_cycle)
        effective_ingredient_score = self.calculate_effective_ingredient_score(
            scoring_data.effective_ingredient_content
        )
        sludge_treatment_rate_score = self.calculate_sludge_treatment_rate_score(
            scoring_data.sludge_treatment_rate
        )
        delivery_timeliness_score = self.calculate_delivery_timeliness_score(scoring_data.delivery_timeliness)
        inventory_capacity_score = self.calculate_inventory_capacity_score(scoring_data.inventory_capacity)
        delivery_capability_score = self.calculate_delivery_capability_score(scoring_data.delivery_capability)
        emergency_supply_score = self.calculate_emergency_supply_score(scoring_data.emergency_supply_capability)
        safety_management_score = self.calculate_safety_management_score(scoring_data.safety_management_level)
        
        total_internal_score = (
            material_unit_price_score +
            transport_cost_score +
            payment_cycle_score +
            effective_ingredient_score +
            sludge_treatment_rate_score +
            delivery_timeliness_score +
            inventory_capacity_score +
            delivery_capability_score +
            emergency_supply_score +
            safety_management_score
        )
        
        score_breakdown = {
            '药剂单价得分(18分)': material_unit_price_score,
            '运输成本得分(9分)': transport_cost_score,
            '付款周期得分(4分)': payment_cycle_score,
            '药剂有效成分含量得分(15分)': effective_ingredient_score,
            '处理污泥效果达标率得分(20分)': sludge_treatment_rate_score,
            '供货及时性得分(10分)': delivery_timeliness_score,
            '库存能力得分(4分)': inventory_capacity_score,
            '交付能力得分(3分)': delivery_capability_score,
            '应急供应能力得分(5分)': emergency_supply_score,
            '安全管理水平得分(12分)': safety_management_score
        }
        
        return SupplierScoringResult(
            supplier_id=scoring_data.supplier_id,
            supplier_name=supplier_name,
            material_unit_price_score=material_unit_price_score,
            transport_cost_score=transport_cost_score,
            payment_cycle_score=payment_cycle_score,
            effective_ingredient_score=effective_ingredient_score,
            sludge_treatment_rate_score=sludge_treatment_rate_score,
            delivery_timeliness_score=delivery_timeliness_score,
            inventory_capacity_score=inventory_capacity_score,
            delivery_capability_score=delivery_capability_score,
            emergency_supply_score=emergency_supply_score,
            safety_management_score=safety_management_score,
            total_internal_score=total_internal_score,
            score_breakdown=score_breakdown
        )

    def batch_calculate_scores(self, scoring_data_list: List[Tuple[SupplierScoringData, str]]) -> List[SupplierScoringResult]:
        if scoring_data_list:
            data_list = [data for data, _ in scoring_data_list]
            self.calculate_data_ranges(data_list)
        
        results = []
        for scoring_data, supplier_name in scoring_data_list:
            result = self.calculate_supplier_score(scoring_data, supplier_name)
            results.append(result)
        return results

    def generate_scoring_report(self, scoring_results: List[SupplierScoringResult]) -> str:
        report = "供应商评分报告\n"
        report += "=" * 80 + "\n\n"
        
        sorted_results = sorted(scoring_results, key=lambda x: x.total_internal_score, reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            report += f"排名 {i}: {result.supplier_name} (ID: {result.supplier_id})\n"
            report += "-" * 80 + "\n"
            report += f"总分: {result.total_internal_score:.2f}/100\n\n"
            
            report += "各项得分明细:\n"
            for criterion, score in result.score_breakdown.items():
                report += f"  {criterion}: {score:.2f}\n"
            
            report += "\n"
        
        return report

    def get_top_suppliers(self, scoring_results: List[SupplierScoringResult], 
                         top_n: int = 5) -> List[SupplierScoringResult]:
        sorted_results = sorted(scoring_results, key=lambda x: x.total_internal_score, reverse=True)
        return sorted_results[:top_n]

    def calculate_score_statistics(self, scoring_results: List[SupplierScoringResult]) -> Dict[str, float]:
        if not scoring_results:
            return {}
        
        scores = [result.total_internal_score for result in scoring_results]
        
        return {
            '平均分': sum(scores) / len(scores),
            '最高分': max(scores),
            '最低分': min(scores),
            '标准差': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
        }