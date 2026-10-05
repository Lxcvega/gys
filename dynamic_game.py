import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from data_model import (
    SupplierCapability, WastewaterType, ChemicalType,
    ChemicalRequirement, SupplierProposal
)
from enum import Enum
class GameStrategy(Enum):
    COOPERATIVE = "合作策略"
    COMPETITIVE = "竞争策略"
    MIXED = "混合策略"

@dataclass
class SupplierGamePosition:
    supplier_id: str
    chemical_type: ChemicalType
    strategy: GameStrategy
    payoff: float
    market_share: float
    competitive_advantage: float
    risk_level: float

@dataclass
class ChemicalOption:
    chemical_type: ChemicalType
    suppliers: List[str]
    total_capacity: float
    average_price: float
    quality_score: float
    urgency_factor: float

@dataclass
class GameResult:
    supplier_id: str
    chemical_type: ChemicalType
    final_rank: int
    game_score: float
    strategy: GameStrategy
    recommended_quantity: float
    expected_payoff: float
    market_position: str

class DynamicGameModule:
    def __init__(self):
        self.wastewater_chemical_priorities = {
            WastewaterType.INDUSTRIAL: {
                ChemicalType.HEAVY_METAL_REMOVER: {'priority': 1.0, 'weight': 0.4},
                ChemicalType.COAGULANT: {'priority': 0.8, 'weight': 0.3},
                ChemicalType.PH_ADJUSTER: {'priority': 0.7, 'weight': 0.3}
            },
            WastewaterType.DOMESTIC: {
                ChemicalType.COAGULANT: {'priority': 1.0, 'weight': 0.35},
                ChemicalType.DISINFECTANT: {'priority': 0.9, 'weight': 0.35},
                ChemicalType.ORGANIC_REMOVER: {'priority': 0.8, 'weight': 0.3}
            },
            WastewaterType.MEDICAL: {
                ChemicalType.DISINFECTANT: {'priority': 1.0, 'weight': 0.4},
                ChemicalType.COAGULANT: {'priority': 0.85, 'weight': 0.3},
                ChemicalType.ORGANIC_REMOVER: {'priority': 0.9, 'weight': 0.3}
            },
            WastewaterType.AGRICULTURAL: {
                ChemicalType.ORGANIC_REMOVER: {'priority': 1.0, 'weight': 0.4},
                ChemicalType.PH_ADJUSTER: {'priority': 0.8, 'weight': 0.3},
                ChemicalType.COAGULANT: {'priority': 0.7, 'weight': 0.3}
            },
            WastewaterType.CHEMICAL: {
                ChemicalType.HEAVY_METAL_REMOVER: {'priority': 1.0, 'weight': 0.35},
                ChemicalType.PH_ADJUSTER: {'priority': 0.9, 'weight': 0.35},
                ChemicalType.ORGANIC_REMOVER: {'priority': 0.85, 'weight': 0.3}
            }
        }
        
        self.urgency_multipliers = {
            '紧急': 1.5,
            '高': 1.2,
            '中': 1.0,
            '低': 0.8
        }

    def map_chemicals_to_wastewater(self, wastewater_type: WastewaterType,
                                    requirements: List[ChemicalRequirement]) -> List[ChemicalOption]:
        chemical_options = []
        
        priority_info = self.wastewater_chemical_priorities.get(wastewater_type, {})
        
        for requirement in requirements:
            chemical_type = requirement.chemical_type
            
            urgency_factor = self.urgency_multipliers.get(requirement.urgency_level, 1.0)
            
            priority = priority_info.get(chemical_type, {}).get('priority', 0.5)
            weight = priority_info.get(chemical_type, {}).get('weight', 0.2)
            
            chemical_options.append(ChemicalOption(
                chemical_type=chemical_type,
                suppliers=[],
                total_capacity=0.0,
                average_price=0.0,
                quality_score=0.0,
                urgency_factor=urgency_factor * priority
            ))
        
        return chemical_options

    def populate_chemical_options(self, chemical_options: List[ChemicalOption],
                                  suppliers: List[SupplierCapability],
                                  proposals: Dict[str, SupplierProposal]) -> List[ChemicalOption]:
        chemical_supplier_map = {}
        
        for supplier in suppliers:
            proposal = proposals.get(supplier.supplier_id)
            if not proposal:
                continue
            
            chemical_type = proposal.chemical_type
            
            if chemical_type not in chemical_supplier_map:
                chemical_supplier_map[chemical_type] = []
            
            chemical_supplier_map[chemical_type].append({
                'supplier': supplier,
                'proposal': proposal
            })
        
        for option in chemical_options:
            if option.chemical_type in chemical_supplier_map:
                option.suppliers = [item['supplier'].supplier_id for item in chemical_supplier_map[option.chemical_type]]
                
                total_capacity = sum(item['supplier'].inventory_capacity for item in chemical_supplier_map[option.chemical_type])
                option.total_capacity = total_capacity
                
                if chemical_supplier_map[option.chemical_type]:
                    avg_price = float(np.mean([item['proposal'].unit_price for item in chemical_supplier_map[option.chemical_type]]))
                    option.average_price = avg_price
                    
                    avg_quality = float(np.mean([item['supplier'].effective_ingredient_content for item in chemical_supplier_map[option.chemical_type]]))
                    option.quality_score = avg_quality
        
        return chemical_options

    def calculate_payoff_matrix(self, suppliers: List[SupplierCapability],
                               proposals: Dict[str, SupplierProposal],
                               chemical_options: List[ChemicalOption]) -> Dict[str, Dict[str, float]]:
        payoff_matrix = {}
        
        for supplier in suppliers:
            proposal = proposals.get(supplier.supplier_id)
            if not proposal:
                continue
            
            supplier_payoffs = {}
            
            for option in chemical_options:
                if supplier.supplier_id not in option.suppliers:
                    supplier_payoffs[option.chemical_type.value] = 0.0
                    continue
                
                price_competitiveness = max(0.0, 1.0 - (proposal.unit_price / option.average_price)) if option.average_price > 0 else 0.5
                quality_advantage = min(1.0, supplier.effective_ingredient_content / option.quality_score) if option.quality_score > 0 else 0.5
                capacity_utilization = supplier.inventory_capacity / option.total_capacity if option.total_capacity > 0 else 0.5
                
                urgency_bonus = option.urgency_factor
                
                payoff = (
                    price_competitiveness * 0.3 +
                    quality_advantage * 0.3 +
                    capacity_utilization * 0.2 +
                    urgency_bonus * 0.2
                )
                
                supplier_payoffs[option.chemical_type.value] = max(0, min(1, payoff))
            
            payoff_matrix[supplier.supplier_id] = supplier_payoffs
        
        return payoff_matrix

    def determine_optimal_strategy(self, supplier: SupplierCapability,
                                  payoff_matrix: Dict[str, Dict[str, float]],
                                  market_conditions: Dict[str, float]) -> GameStrategy:
        supplier_payoffs = payoff_matrix.get(supplier.supplier_id, {})
        
        if not supplier_payoffs:
            return GameStrategy.MIXED
        
        avg_payoff = np.mean(list(supplier_payoffs.values()))
        max_payoff = max(supplier_payoffs.values())
        min_payoff = min(supplier_payoffs.values())
        
        payoff_variance = max_payoff - min_payoff
        
        market_competition = market_conditions.get('competition_level', 0.5)
        
        if avg_payoff > 0.7 and payoff_variance < 0.2:
            return GameStrategy.COOPERATIVE
        elif market_competition > 0.7 and payoff_variance > 0.3:
            return GameStrategy.COMPETITIVE
        else:
            return GameStrategy.MIXED

    def calculate_nash_equilibrium(self, payoff_matrix: Dict[str, Dict[str, float]],
                                   suppliers: List[SupplierCapability]) -> Dict[str, SupplierGamePosition]:
        nash_positions = {}
        
        supplier_ids = [s.supplier_id for s in suppliers]
        chemical_types = list(next(iter(payoff_matrix.values())).keys()) if payoff_matrix else []
        
        for supplier in suppliers:
            supplier_id = supplier.supplier_id
            supplier_payoffs = payoff_matrix.get(supplier_id, {})
            
            if not supplier_payoffs:
                continue
            
            best_chemical = max(supplier_payoffs.items(), key=lambda x: x[1])
            best_chemical_type = best_chemical[0]
            best_payoff = best_chemical[1]
            
            chemical_type = None
            for ct in ChemicalType:
                if ct.value == best_chemical_type:
                    chemical_type = ct
                    break
            
            if chemical_type:
                market_share = supplier.inventory_capacity / sum(s.inventory_capacity for s in suppliers)
                
                competitive_advantage = best_payoff * (1.0 + supplier.emergency_supply_capability / 5.0 * 0.2)
                
                risk_level = 1.0 - (supplier.delivery_capability / 5.0) * 0.5 - (supplier.safety_management_level / 5.0) * 0.5
                
                num_suppliers = len(suppliers)
                competition_level = min(1.0, num_suppliers / 10.0)
                strategy = self.determine_optimal_strategy(
                    supplier, payoff_matrix, {'competition_level': competition_level}
                )
                
                nash_positions[supplier_id] = SupplierGamePosition(
                    supplier_id=supplier_id,
                    chemical_type=chemical_type,
                    strategy=strategy,
                    payoff=best_payoff,
                    market_share=market_share,
                    competitive_advantage=competitive_advantage,
                    risk_level=risk_level
                )
        
        return nash_positions

    def dynamic_ranking(self, suppliers: List[SupplierCapability],
                       proposals: Dict[str, SupplierProposal],
                       chemical_options: List[ChemicalOption],
                       nash_positions: Dict[str, SupplierGamePosition]) -> List[GameResult]:
        game_results = []
        
        for supplier in suppliers:
            supplier_id = supplier.supplier_id
            position = nash_positions.get(supplier_id)
            
            if not position:
                continue
            
            proposal = proposals.get(supplier_id)
            if not proposal:
                continue
            
            chemical_option = next((opt for opt in chemical_options 
                                   if opt.chemical_type == position.chemical_type), None)
            
            if not chemical_option:
                continue
            
            base_score = position.payoff * 0.4
            strategy_bonus = {
                GameStrategy.COOPERATIVE: 0.1,
                GameStrategy.COMPETITIVE: 0.05,
                GameStrategy.MIXED: 0.0
            }.get(position.strategy, 0.0)
            
            market_advantage = position.competitive_advantage * 0.2
            risk_penalty = position.risk_level * 0.1
            urgency_bonus = chemical_option.urgency_factor * 0.15
            
            game_score = base_score + strategy_bonus + market_advantage - risk_penalty + urgency_bonus
            game_score = max(0, min(1, game_score))
            
            recommended_quantity = min(
                proposal.offered_quantity,
                supplier.inventory_capacity * 0.8
            )
            
            market_position = self._determine_market_position(game_score, position.market_share)
            
            game_results.append(GameResult(
                supplier_id=supplier_id,
                chemical_type=position.chemical_type,
                final_rank=0,
                game_score=game_score,
                strategy=position.strategy,
                recommended_quantity=recommended_quantity,
                expected_payoff=position.payoff,
                market_position=market_position
            ))
        
        sorted_results = sorted(game_results, key=lambda x: x.game_score, reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            result.final_rank = i
        
        return sorted_results

    def _determine_market_position(self, game_score: float, market_share: float) -> str:
        if game_score > 0.8 and market_share > 0.3:
            return "市场领导者"
        elif game_score > 0.7:
            return "强力竞争者"
        elif game_score > 0.5:
            return "稳健参与者"
        elif game_score > 0.3:
            return "潜力供应商"
        else:
            return "边缘供应商"

    def allocate_quantities(self, game_results: List[GameResult],
                           requirements: List[ChemicalRequirement]) -> Dict[str, float]:
        allocation = {}
        
        chemical_requirements = {}
        for req in requirements:
            chemical_requirements[req.chemical_type] = req.required_quantity
        
        for result in game_results:
            chemical_type = result.chemical_type
            required = chemical_requirements.get(chemical_type, 0)
            
            if required > 0:
                allocated = min(result.recommended_quantity, required)
                allocation[result.supplier_id] = allocated
                chemical_requirements[chemical_type] -= allocated
        
        return allocation

    def run_dynamic_game(self, suppliers: List[SupplierCapability],
                        proposals: Dict[str, SupplierProposal],
                        requirements: List[ChemicalRequirement],
                        wastewater_type: WastewaterType) -> Tuple[List[GameResult], Dict[str, float]]:
        chemical_options = self.map_chemicals_to_wastewater(wastewater_type, requirements)
        chemical_options = self.populate_chemical_options(chemical_options, suppliers, proposals)
        
        payoff_matrix = self.calculate_payoff_matrix(suppliers, proposals, chemical_options)
        
        nash_positions = self.calculate_nash_equilibrium(payoff_matrix, suppliers)
        
        game_results = self.dynamic_ranking(suppliers, proposals, chemical_options, nash_positions)
        
        quantity_allocation = self.allocate_quantities(game_results, requirements)
        
        return game_results, quantity_allocation

    def generate_game_report(self, game_results: List[GameResult],
                            quantity_allocation: Dict[str, float]) -> str:
        report = "动态博弈排序报告\n"
        report += "=" * 60 + "\n\n"
        
        report += f"参与供应商数量: {len(game_results)}\n"
        report += f"总分配数量: {sum(quantity_allocation.values()):.2f}\n\n"
        
        strategy_counts = {}
        for result in game_results:
            strategy = result.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        report += "策略分布:\n"
        for strategy, count in strategy_counts.items():
            report += f"• {strategy}: {count} 家供应商\n"
        
        report += "\n供应商排序结果:\n"
        report += "-" * 60 + "\n"
        
        for result in game_results:
            report += f"{result.final_rank}. {result.supplier_id}\n"
            report += f"   药水类型: {result.chemical_type.value}\n"
            report += f"   博弈评分: {result.game_score:.4f}\n"
            report += f"   策略: {result.strategy.value}\n"
            report += f"   市场定位: {result.market_position}\n"
            report += f"   推荐数量: {quantity_allocation.get(result.supplier_id, 0):.2f}\n"
            report += f"   预期收益: {result.expected_payoff:.4f}\n\n"
        
        return report