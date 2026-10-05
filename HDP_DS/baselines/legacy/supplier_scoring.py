from typing import Dict, List, Tuple
from .data_model import SupplierScoringData, SupplierScoringResult

class SupplierScoringModule:
    def __init__(self):
        # 评分标准（依据 评分标准.xlsx）
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
        # 满分之和 = 18+9+4+15+20+10+4+3+5+12 = 100（直接百分制）
    
    def calculate_data_ranges(self, scoring_data_list: List[SupplierScoringData]):
        """仅打印数据概况，评分使用固定行业标准范围"""
        if not scoring_data_list:
            return
        
        material_prices = [data.material_unit_price for data in scoring_data_list]
        # 查找最高最低单价用于药剂单价评分
        self.price_max = max(material_prices)
        self.price_min = min(material_prices)
        
        print("\n=== 评分数据概况（使用固定行业标准范围） ===")
        print(f"药剂单价范围: {self.price_min:.2f} - {self.price_max:.2f} 元/吨（使用实际最大最小值）")
        print(f"运输成本: 固定范围 177-401 元/吨")
        print(f"付款周期: 固定范围 7-60 天")
        print(f"药剂有效成分含量: 固定范围 85-98%")
        print(f"处理污泥效果达标率: 固定范围 80-99%")
        print(f"供货及时性: 固定范围 85-100%")
        print(f"库存能力: 固定范围 100-1000 吨")
        print(f"交付能力: 固定范围 1-5 级")
        print(f"应急供应能力: 固定范围 3-5 级")
        print(f"安全管理水平: 固定范围 3-5 级")

    def calculate_material_unit_price_score(self, unit_price: float) -> float:
        # 负向指标：使用数据实际最大最小值
        if not hasattr(self, 'price_max') or self.price_max == self.price_min:
            return self.max_scores['material_unit_price']
        score = 18 * (self.price_max - unit_price) / (self.price_max - self.price_min)
        return max(0, min(18, score))

    def calculate_transport_cost_score(self, transport_cost: float) -> float:
        # 负向指标：固定范围 177-401
        score = 9 * (401 - transport_cost) / 224
        return max(0, min(9, score))

    def calculate_payment_cycle_score(self, payment_cycle: float) -> float:
        # 负向指标：固定范围 7-60
        score = 4 * (60 - payment_cycle) / 53
        return max(0, min(4, score))

    def calculate_effective_ingredient_score(self, effective_ingredient: float) -> float:
        # 正向指标：固定范围 85-98
        score = 15 * (effective_ingredient - 85) / 13
        return max(0, min(15, score))

    def calculate_sludge_treatment_rate_score(self, rate: float) -> float:
        # 正向指标：固定范围 80-99
        score = 20 * (rate - 80) / 19
        return max(0, min(20, score))

    def calculate_delivery_timeliness_score(self, timeliness: float) -> float:
        # 正向指标：固定范围 85-100
        score = 10 * (timeliness - 85) / 15
        return max(0, min(10, score))

    def calculate_inventory_capacity_score(self, inventory_capacity: float) -> float:
        # 正向指标：固定范围 100-1000
        score = 4 * (inventory_capacity - 100) / 900
        return max(0, min(4, score))

    def calculate_delivery_capability_score(self, delivery_capability: int) -> float:
        # 正向指标：固定范围 1-5
        score = 3 * (delivery_capability - 1) / 4
        return max(0, min(3, score))

    def calculate_emergency_supply_score(self, emergency_level: int) -> float:
        # 正向指标：固定范围 3-5
        score = 5 * (emergency_level - 3) / 2
        return max(0, min(5, score))

    def calculate_safety_management_score(self, safety_level: int) -> float:
        # 正向指标：固定范围 3-5
        score = 12 * (safety_level - 3) / 2
        return max(0, min(12, score))

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
        
        # 满分100分，直接加总（不需要归一化）
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