import numpy as np
from typing import Dict, List
from data_model import (
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
        
        geometric_mean = float(np.prod(expert_scores) ** (1/n))
        return geometric_mean

    def update_criteria_weights(self, expert_judgments: Dict[str, List[float]]) -> Dict[str, float]:
        updated_weights = {}
        for criterion, scores in expert_judgments.items():
            updated_weights[criterion] = self.calculate_fuzzy_weight(scores)
        
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            for criterion in updated_weights:
                updated_weights[criterion] /= total_weight
        
        self.criteria_weights = updated_weights
        return updated_weights

    def classify_wastewater(self, characteristics: WastewaterCharacteristics) -> WastewaterType:
        scores = {}
        
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
        if criterion in ('effective_ingredient_content', 'delivery_timeliness', 'sludge_treatment_rate'):
            return min(1.0, value / 100.0)
        if criterion in ('delivery_capability', 'emergency_supply_capability', 'safety_management_level'):
            return min(1.0, value / 5.0)
        if criterion in ('material_unit_price',):
            return min(1.0, value / 5000.0)
        if criterion in ('transport_cost',):
            return min(1.0, value / 500.0)
        if criterion in ('payment_cycle',):
            return max(0.0, 1.0 - value / 120.0)
        if criterion in ('inventory_capacity',):
            return min(1.0, value / 10000.0)
        return min(max(value, 0), 1)

    def calculate_fuzzy_score(self, value: float, criterion: str) -> FuzzyScore:
        normalized_value = self._normalize_by_criterion(value, criterion)
        
        memberships = {}
        for term, (a, b, c) in self.linguistic_terms.items():
            mu = self._triangular_membership(normalized_value, a, b, c)
            memberships[term] = mu
        
        best_term = max(memberships, key=lambda k: memberships[k])
        membership_value = memberships[best_term]
        
        numerator = 0.0
        denominator = 0.0
        for term, mu in memberships.items():
            if mu > 0:
                a, b, c = self.linguistic_terms[term]
                centroid = (a + b + c) / 3.0
                numerator += mu * centroid
                denominator += mu
        defuzzified = numerator / denominator if denominator > 0 else normalized_value
        
        return FuzzyScore(
            linguistic_term=best_term,
            membership_value=round(membership_value, 4),
            defuzzified_value=round(defuzzified, 4)
        )

    @staticmethod
    def _triangular_membership(x: float, a: float, b: float, c: float) -> float:
        """三角隶属度函数: trimf(x; a, b, c)"""
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a) if b > a else 1.0
        if b < x < c:
            return (c - x) / (c - b) if c > b else 1.0
        return 0.0

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