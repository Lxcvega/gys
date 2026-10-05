import numpy as np
from typing import Dict, List
from .data_model import (
    WastewaterType, ChemicalType, SupplierCapability,
    WastewaterCharacteristics, FuzzyScore
)

class FuzzyDecisionModule:
    def __init__(self):
        self.linguistic_terms = {
            '极差': (0, 0, 0.2),
            '差': (0, 0.2, 0.4),
            '一般': (0.2, 0.4, 0.6),
            '良': (0.4, 0.6, 0.8),
            '优': (0.6, 0.8, 1.0),
            '极优': (0.8, 1.0, 1.0)
        }
        
        self.criteria_weights = {
            'material_unit_price': 0.136,
            'transport_cost': 0.091,
            'payment_cycle': 0.091,
            'effective_ingredient_content': 0.182,
            'delivery_timeliness': 0.136,
            'inventory_capacity': 0.045,
            'delivery_capability': 0.091,
            'emergency_supply_capability': 0.091,
            'safety_management_level': 0.137
        }
        
        self.wastewater_chemical_mapping = {
            WastewaterType.INDUSTRIAL: [
                ChemicalType.HEAVY_METAL_REMOVER,
                ChemicalType.COAGULANT,
                ChemicalType.PH_ADJUSTER
            ],
            WastewaterType.DOMESTIC: [
                ChemicalType.COAGULANT,
                ChemicalType.DISINFECTANT,
                ChemicalType.ORGANIC_REMOVER
            ],
            WastewaterType.MEDICAL: [
                ChemicalType.DISINFECTANT,
                ChemicalType.COAGULANT,
                ChemicalType.ORGANIC_REMOVER
            ],
            WastewaterType.AGRICULTURAL: [
                ChemicalType.ORGANIC_REMOVER,
                ChemicalType.PH_ADJUSTER,
                ChemicalType.COAGULANT
            ],
            WastewaterType.CHEMICAL: [
                ChemicalType.HEAVY_METAL_REMOVER,
                ChemicalType.PH_ADJUSTER,
                ChemicalType.ORGANIC_REMOVER
            ]
        }

    def calculate_fuzzy_weight(self, expert_scores: List[float]) -> float:
        n = len(expert_scores)
        if n == 0:
            return 0.0
        
        # 几何平均作为该准则的综合专家评分
        geometric_mean = float(np.prod(expert_scores) ** (1/n))
        return geometric_mean

    def update_criteria_weights(self, expert_judgments: Dict[str, List[float]]) -> Dict[str, float]:
        updated_weights = {}
        for criterion, scores in expert_judgments.items():
            updated_weights[criterion] = self.calculate_fuzzy_weight(scores)
        
        # 归一化使权重之和为1
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            for criterion in updated_weights:
                updated_weights[criterion] /= total_weight
        
        self.criteria_weights = updated_weights
        return updated_weights

    def classify_wastewater(self, characteristics: WastewaterCharacteristics) -> WastewaterType:
        scores = {}
        
        # 输入重归一化：将原始量纲映射到 [0,1] 后再与阈值比较
        # 典型范围：重金属 0~100 mg/L, 有机物 0~500 mg/L, 毒性 0~100
        norm_heavy_metal = min(1.0, characteristics.heavy_metal_content / 100.0)
        norm_organic = min(1.0, characteristics.organic_content / 500.0)
        norm_toxicity = min(1.0, characteristics.toxicity_level / 100.0)
        
        if norm_heavy_metal > 0.5:
            scores[WastewaterType.INDUSTRIAL] = scores.get(WastewaterType.INDUSTRIAL, 0) + 0.3
            scores[WastewaterType.CHEMICAL] = scores.get(WastewaterType.CHEMICAL, 0) + 0.3
        
        if norm_organic > 0.6:
            scores[WastewaterType.DOMESTIC] = scores.get(WastewaterType.DOMESTIC, 0) + 0.25
            scores[WastewaterType.AGRICULTURAL] = scores.get(WastewaterType.AGRICULTURAL, 0) + 0.25
        
        if norm_toxicity > 0.7:
            scores[WastewaterType.MEDICAL] = scores.get(WastewaterType.MEDICAL, 0) + 0.35
            scores[WastewaterType.CHEMICAL] = scores.get(WastewaterType.CHEMICAL, 0) + 0.25
        
        if 6.5 <= characteristics.ph_value <= 8.5:
            scores[WastewaterType.DOMESTIC] = scores.get(WastewaterType.DOMESTIC, 0) + 0.15
        
        if characteristics.cod_level > 500:
            scores[WastewaterType.INDUSTRIAL] = scores.get(WastewaterType.INDUSTRIAL, 0) + 0.2
        
        if not scores:
            return WastewaterType.DOMESTIC
        
        return max(scores.items(), key=lambda x: x[1])[0]

    def _normalize_by_criterion(self, value: float, criterion: str) -> float:
        """按字段类型将原始值归一化到 [0,1] 区间"""
        # 百分比类（0-100 → 0-1）
        if criterion in ('effective_ingredient_content', 'delivery_timeliness', 'sludge_treatment_rate'):
            return min(1.0, value / 100.0)
        # 等级类（1-5 → 0-1）
        if criterion in ('delivery_capability', 'emergency_supply_capability', 'safety_management_level'):
            return min(1.0, value / 5.0)
        # 价格类（元/吨），假设行业上限 5000 元/吨
        if criterion in ('material_unit_price',):
            return min(1.0, value / 5000.0)
        # 运输成本，假设上限 500 元/吨
        if criterion in ('transport_cost',):
            return min(1.0, value / 500.0)
        # 付款周期，假设上限 120 天
        if criterion in ('payment_cycle',):
            return max(0.0, 1.0 - value / 120.0)  # 越短越好
        # 库存能力，假设上限 10000 吨
        if criterion in ('inventory_capacity',):
            return min(1.0, value / 10000.0)
        # 未知字段，直接裁剪
        return min(max(value, 0), 1)

    def calculate_fuzzy_score(self, value: float, criterion: str) -> FuzzyScore:
        normalized_value = self._normalize_by_criterion(value, criterion)
        
        if normalized_value <= 0.2:
            linguistic_term = '极差'
            membership_value = 1.0 - (normalized_value / 0.2)
        elif normalized_value <= 0.4:
            linguistic_term = '差'
            membership_value = 1.0 - abs(normalized_value - 0.3) / 0.1
        elif normalized_value <= 0.6:
            linguistic_term = '一般'
            membership_value = 1.0 - abs(normalized_value - 0.5) / 0.1
        elif normalized_value <= 0.8:
            linguistic_term = '良'
            membership_value = 1.0 - abs(normalized_value - 0.7) / 0.1
        else:
            linguistic_term = '优'
            membership_value = normalized_value - 0.8
        
        if membership_value < 0:
            membership_value = 0
        
        return FuzzyScore(
            linguistic_term=linguistic_term,
            membership_value=membership_value,
            defuzzified_value=normalized_value
        )

    def evaluate_supplier_capability(self, capability: SupplierCapability) -> Dict[str, FuzzyScore]:
        fuzzy_scores = {}
        
        criteria_mapping = {
            'material_unit_price': capability.material_unit_price,
            'transport_cost': capability.transport_cost,
            'payment_cycle': capability.payment_cycle,
            'effective_ingredient_content': capability.effective_ingredient_content,
            'delivery_timeliness': capability.delivery_timeliness,
            'inventory_capacity': capability.inventory_capacity,
            'delivery_capability': capability.delivery_capability,
            'emergency_supply_capability': capability.emergency_supply_capability,
            'safety_management_level': capability.safety_management_level
        }
        
        for criterion, value in criteria_mapping.items():
            fuzzy_scores[criterion] = self.calculate_fuzzy_score(value, criterion)
        
        return fuzzy_scores

    def calculate_weighted_fuzzy_score(self, fuzzy_scores: Dict[str, FuzzyScore]) -> float:
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, fuzzy_score in fuzzy_scores.items():
            weight = self.criteria_weights.get(criterion, 0.1)
            total_score += fuzzy_score.defuzzified_value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def get_recommended_chemicals(self, wastewater_type: WastewaterType) -> List[ChemicalType]:
        return self.wastewater_chemical_mapping.get(wastewater_type, [])