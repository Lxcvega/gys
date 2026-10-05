import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from .data_model import (
    SupplierCapability, SupplierProposal, SupplierScoringData,
    ChemicalType
)
@dataclass
class ExcelSupplierData:
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
    contact_person: str = ''
    phone: str = ''
    address: str = ''
    quality_certification: Optional[List[str]] = None
    environmental_certification: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.quality_certification is None:
            self.quality_certification = []
        if self.environmental_certification is None:
            self.environmental_certification = []

class ExcelDataReader:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.supplier_data = []
        
    def read_excel_data(self) -> List[ExcelSupplierData]:
        """读取Excel文件中的供应商数据"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=0)
            
            required_columns = [
                '供应商编号', '药剂单价(元/吨)', '运输成本(元/吨)',
                '付款周期(天)', '药剂有效成分含量(%)',
                '供货及时性(%)', '库存能力(吨)', '交付能力(1-5级)',
                '应急供应能力(1-5级)', '安全管理水平(1-5级)'
            ]
            
            optional_columns = ['供应商名称', '联系人', '电话', '地址', '质量认证', '环境认证', '处理污泥效果达标率(%)']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告：Excel文件中缺少以下必需列: {missing_columns}")
                print(f"可用的列: {list(df.columns)}")
                return []
            
            # ── 数据预处理：缺失值填充 + 异常值修正 ──
            try:
                from .data_processor import DataProcessor
                dp = DataProcessor()
                df, _ = dp.process_dataframe(df)
            except ImportError:
                print("  注意: data_processor 模块未加载，跳过数据预处理")
            except Exception as e:
                print(f"  警告: 数据预处理失败 ({e})，继续使用原始数据")
            # ──────────────────────────────────────────
            
            self.supplier_data = []
            
            for row_idx, (idx, row) in enumerate(df.iterrows()):
                try:
                    supplier_data = ExcelSupplierData(
                        supplier_id=str(row['供应商编号']),
                        supplier_name=str(row.get('供应商名称', f"供应商{row['供应商编号']}")),
                        material_unit_price=float(row['药剂单价(元/吨)']),
                        transport_cost=float(row['运输成本(元/吨)']),
                        payment_cycle=float(row['付款周期(天)']),
                        effective_ingredient_content=float(row['药剂有效成分含量(%)']),
                        delivery_timeliness=float(row['供货及时性(%)']),
                        inventory_capacity=float(row['库存能力(吨)']),
                        delivery_capability=int(row['交付能力(1-5级)']),
                        emergency_supply_capability=int(row['应急供应能力(1-5级)']),
                        safety_management_level=int(row['安全管理水平(1-5级)']),
                        sludge_treatment_rate=float(row.get('处理污泥效果达标率(%)', 0)),
                        contact_person=str(row.get('联系人', '')),
                        phone=str(row.get('电话', '')),
                        address=str(row.get('地址', '')),
                        quality_certification=self._parse_certifications(row.get('质量认证', '')),
                        environmental_certification=self._parse_certifications(row.get('环境认证', ''))
                    )
                    self.supplier_data.append(supplier_data)
                    
                except Exception as e:
                    print(f"警告：第{row_idx+2}行数据解析失败: {e}")
                    continue
            
            print(f"成功读取 {len(self.supplier_data)} 条供应商数据")
            
            return self.supplier_data
            
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return []
    
    def _parse_certifications(self, cert_string: str) -> List[str]:
        """解析认证信息字符串"""
        if pd.isna(cert_string) or cert_string == '':
            return []
        
        if isinstance(cert_string, str):
            return [cert.strip() for cert in cert_string.split(';') if cert.strip()]
        
        return [str(cert_string)]
    
    def convert_to_supplier_capability(self, excel_data: ExcelSupplierData) -> SupplierCapability:
        """将Excel数据转换为供应商能力对象"""
        return SupplierCapability(
            supplier_id=excel_data.supplier_id,
            supplier_name=excel_data.supplier_name,
            material_unit_price=excel_data.material_unit_price,
            transport_cost=excel_data.transport_cost,
            payment_cycle=excel_data.payment_cycle,
            effective_ingredient_content=excel_data.effective_ingredient_content,
            delivery_timeliness=excel_data.delivery_timeliness,
            inventory_capacity=excel_data.inventory_capacity,
            delivery_capability=excel_data.delivery_capability,
            emergency_supply_capability=excel_data.emergency_supply_capability,
            safety_management_level=excel_data.safety_management_level,
            sludge_treatment_rate=excel_data.sludge_treatment_rate
        )
    
    def convert_to_supplier_scoring_data(self, excel_data: ExcelSupplierData) -> SupplierScoringData:
        """将Excel数据转换为供应商评分数据对象"""
        return SupplierScoringData(
            supplier_id=excel_data.supplier_id,
            material_unit_price=excel_data.material_unit_price,
            transport_cost=excel_data.transport_cost,
            payment_cycle=excel_data.payment_cycle,
            effective_ingredient_content=excel_data.effective_ingredient_content,
            delivery_timeliness=excel_data.delivery_timeliness,
            inventory_capacity=excel_data.inventory_capacity,
            delivery_capability=excel_data.delivery_capability,
            emergency_supply_capability=excel_data.emergency_supply_capability,
            safety_management_level=excel_data.safety_management_level,
            sludge_treatment_rate=excel_data.sludge_treatment_rate
        )
    
    def convert_to_supplier_proposal(self, excel_data: ExcelSupplierData, 
                                    chemical_type: ChemicalType,
                                    required_quantity: float) -> SupplierProposal:
        """将Excel数据转换为供应商提案对象"""
        return SupplierProposal(
            supplier_id=excel_data.supplier_id,
            chemical_type=chemical_type,
            offered_quantity=min(excel_data.inventory_capacity, required_quantity * 1.5),
            unit_price=excel_data.material_unit_price,
            delivery_time=7,
            quality_certification=excel_data.quality_certification or [],
            environmental_certification=excel_data.environmental_certification or [],
            additional_services=['技术支持', '售后服务']
        )
    
    def get_all_suppliers(self) -> List[SupplierCapability]:
        """获取所有供应商能力对象"""
        return [self.convert_to_supplier_capability(data) for data in self.supplier_data]
    
    def get_all_scoring_data(self) -> List[SupplierScoringData]:
        """获取所有供应商评分数据对象"""
        return [self.convert_to_supplier_scoring_data(data) for data in self.supplier_data]
    
    def generate_data_summary(self) -> Dict:
        """生成数据摘要"""
        if not self.supplier_data:
            return {}
        
        summary = {
            'total_suppliers': len(self.supplier_data),
            'avg_material_price': np.mean([d.material_unit_price for d in self.supplier_data]),
            'avg_transport_cost': np.mean([d.transport_cost for d in self.supplier_data]),
            'avg_payment_cycle': np.mean([d.payment_cycle for d in self.supplier_data]),
            'avg_effective_ingredient': np.mean([d.effective_ingredient_content for d in self.supplier_data]),
            'avg_delivery_timeliness': np.mean([d.delivery_timeliness for d in self.supplier_data]),
            'avg_inventory_capacity': np.mean([d.inventory_capacity for d in self.supplier_data])
        }
        
        return summary
    
    def print_data_summary(self):
        """打印数据摘要"""
        summary = self.generate_data_summary()
        
        print("\n" + "="*60)
        print("供应商数据摘要")
        print("="*60)
        print(f"总供应商数量: {summary['total_suppliers']}")
        
        print(f"\n平均药剂单价: {summary['avg_material_price']:.2f} 元/吨")
        print(f"平均运输成本: {summary['avg_transport_cost']:.2f} 元/吨")
        print(f"平均付款周期: {summary['avg_payment_cycle']:.2f} 天")
        print(f"平均药剂有效成分含量: {summary['avg_effective_ingredient']:.2f} %")
        print(f"平均供货及时性: {summary['avg_delivery_timeliness']:.2f} %")
        print(f"平均库存能力: {summary['avg_inventory_capacity']:.2f} 吨")
        print("="*60)


def create_mock_excel_template(output_path: str = "供应商数据模板.xlsx"):
    """创建Excel数据模板文件"""
    template_data = {
        '供应商ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
        '供应商名称': ['化工药剂有限公司', '环保科技股份', '水处理材料厂', '净化技术集团', '绿色化工公司'],
        '联系人': ['张经理', '李总监', '王厂长', '赵董事长', '刘主任'],
        '电话': ['13800138001', '13800138002', '13800138003', '13800138004', '13800138005'],
        '地址': ['北京市朝阳区', '上海市浦东新区', '广州市天河区', '深圳市南山区', '杭州市西湖区'],
        '药剂类型': ['聚合氯化铝', '聚丙烯酰胺', '三氯化铁', '次氯酸钠', '活性炭'],
        '材料单价': [2800.0, 15000.0, 3200.0, 1800.0, 8500.0],
        '运输成本': [150.0, 200.0, 180.0, 120.0, 250.0],
        '运输时间': [3.0, 5.0, 4.0, 2.0, 6.0],
        '药剂浓度符合度': [0.95, 0.98, 0.92, 0.96, 0.94],
        '批次完成率': [0.97, 0.99, 0.95, 0.98, 0.96],
        '交付及时性': [0.94, 0.98, 0.93, 0.97, 0.95],
        '供应能力': [5000.0, 3000.0, 6000.0, 8000.0, 4000.0],
        '应急供应能力': [3, 2, 4, 5, 3],
        '安全管理水平': [4, 5, 4, 5, 4],
        '生产能力': [5000.0, 3000.0, 6000.0, 8000.0, 4000.0],
        '质量评分': [0.85, 0.92, 0.88, 0.90, 0.87],
        '交付可靠性': [0.90, 0.95, 0.88, 0.93, 0.91],
        '价格竞争力': [0.82, 0.78, 0.85, 0.80, 0.83],
        '技术能力': [0.86, 0.94, 0.89, 0.91, 0.88],
        '环境合规性': [0.92, 0.96, 0.90, 0.95, 0.93],
        '财务稳定性': [0.88, 0.93, 0.87, 0.92, 0.89],
        '客户满意度': [0.90, 0.95, 0.89, 0.94, 0.91],
        '创新能力': [0.84, 0.91, 0.86, 0.89, 0.85],
        '质量认证': ['ISO9001', 'ISO9001;ISO14001', 'ISO9001', 'ISO9001;ISO14001;OHSAS18001', 'ISO9001;ISO14001'],
        '环境认证': ['ISO14001', 'ISO14001;绿色认证', 'ISO14001', 'ISO14001;绿色认证;环保认证', 'ISO14001;绿色认证']
    }
    
    df = pd.DataFrame(template_data)
    df.to_excel(output_path, index=False, sheet_name='供应商数据')
    print(f"Excel模板文件已创建: {output_path}")
    return output_path