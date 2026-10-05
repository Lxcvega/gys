from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class WastewaterType(Enum):
    INDUSTRIAL = "工业污水"
    DOMESTIC = "生活污水"
    MEDICAL = "医疗污水"
    AGRICULTURAL = "农业污水"
    CHEMICAL = "化工污水"

class ChemicalType(Enum):
    COAGULANT = "絮凝剂"
    DISINFECTANT = "消毒剂"
    PH_ADJUSTER = "pH调节剂"
    HEAVY_METAL_REMOVER = "重金属去除剂"
    ORGANIC_REMOVER = "有机物去除剂"
    POLYALUMINUM_CHLORIDE = "聚合氯化铝"
    POLYACRYLAMIDE = "聚丙烯酰胺"
    FERRIC_CHLORIDE = "三氯化铁"
    ALUMINUM_SULFATE = "硫酸铝"
    SODIUM_HYPOCHLORITE = "次氯酸钠"
    CALCIUM_HYPOCHLORITE = "漂白粉"
    HYDROGEN_PEROXIDE = "双氧水"
    SODIUM_BISULFITE = "亚硫酸钠"
    ACTIVATED_CARBON = "活性炭"
    LIME = "生石灰"
    SODA_ASH = "纯碱"
    FLOCCULANT = "助凝剂"
    OXIDIZING_AGENT = "氧化剂"
    REDUCING_AGENT = "还原剂"

@dataclass
class GovernmentEmissionConstraint:
    max_daily_emission: float
    max_monthly_emission: float
    max_annual_emission: float
    pollutant_type: str
    penalty_rate: float

@dataclass
class SupplierCapability:
    supplier_id: str
    supplier_name: str
    material_unit_price: float
    transport_cost: float
    payment_cycle: float
    effective_ingredient_content: float
    delivery_timeliness: float
    inventory_capacity: float
    delivery_capability: int
    emergency_supply_capability: int
    safety_management_level: int
    sludge_treatment_rate: float = 0.0

@dataclass
class FuzzyScore:
    linguistic_term: str
    membership_value: float
    defuzzified_value: float

@dataclass
class WastewaterCharacteristics:
    wastewater_type: WastewaterType
    ph_value: float
    cod_level: float
    bod_level: float
    suspended_solids: float
    heavy_metal_content: float
    organic_content: float
    toxicity_level: float

@dataclass
class ChemicalRequirement:
    chemical_type: ChemicalType
    required_quantity: float
    quality_standard: str
    urgency_level: str
    budget_constraint: float

@dataclass
class SupplierProposal:
    supplier_id: str
    chemical_type: ChemicalType
    offered_quantity: float
    unit_price: float
    delivery_time: int
    quality_certification: List[str]
    environmental_certification: List[str]
    additional_services: List[str]

@dataclass
class SupplierPoolMember:
    supplier_id: str
    pareto_rank: int
    crowding_distance: float
    objectives: Dict[str, float]
    dominance_count: int
    dominated_solutions: List['SupplierPoolMember']

@dataclass
class SelectionResult:
    supplier_id: str
    supplier_name: str
    final_score: float
    is_selected: bool
    selection_reason: str
    risk_assessment: float
    expected_benefit: float
    contract_terms: Dict[str, Any]

@dataclass
class SupplierScoringData:
    supplier_id: str
    material_unit_price: float
    transport_cost: float
    payment_cycle: float
    effective_ingredient_content: float
    delivery_timeliness: float
    inventory_capacity: float
    delivery_capability: int
    emergency_supply_capability: int
    safety_management_level: int
    sludge_treatment_rate: float = 0.0

@dataclass
class SupplierScoringResult:
    supplier_id: str
    supplier_name: str
    material_unit_price_score: float
    transport_cost_score: float
    payment_cycle_score: float
    effective_ingredient_score: float
    sludge_treatment_rate_score: float
    delivery_timeliness_score: float
    inventory_capacity_score: float
    delivery_capability_score: float
    emergency_supply_score: float
    safety_management_score: float
    total_internal_score: float
    score_breakdown: Dict[str, float]