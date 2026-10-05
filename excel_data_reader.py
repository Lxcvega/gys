import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from data_model import (
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
    _GRADE_COLUMNS = ('交付能力(1-5级)', '应急供应能力(1-5级)', '安全管理水平(1-5级)')
    _PERCENT_COLUMNS = ('药剂有效成分含量(%)', '供货及时性(%)', '处理污泥效果达标率(%)')
    _NUMERIC_COLUMNS = (
        '药剂单价(元/吨)', '运输成本(元/吨)', '付款周期(天)',
        '药剂有效成分含量(%)', '供货及时性(%)', '库存能力(吨)',
        '交付能力(1-5级)', '应急供应能力(1-5级)', '安全管理水平(1-5级)',
        '处理污泥效果达标率(%)',
    )

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.supplier_data = []
        
    def read_excel_data(self) -> List[ExcelSupplierData]:
        # Clear before any I/O so failed subsequent reads cannot expose stale data.
        self.supplier_data = []
        try:
            # Preserve literal "nan"/"NULL" text for validation; only actual blanks
            # (or whitespace cells) qualify for the documented missing-value policy.
            df = pd.read_excel(self.excel_path, sheet_name=0, keep_default_na=False)
            
            required_columns = [
                '供应商编号', '药剂单价(元/吨)', '运输成本(元/吨)',
                '付款周期(天)', '药剂有效成分含量(%)',
                '供货及时性(%)', '库存能力(吨)', '交付能力(1-5级)',
                '应急供应能力(1-5级)', '安全管理水平(1-5级)'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告：Excel文件中缺少以下必需列: {missing_columns}")
                print(f"可用的列: {list(df.columns)}")
                return []

            if df.empty:
                print('警告：Excel文件中没有供应商数据')
                return []

            ids = df['供应商编号'].map(self._normalize_supplier_id)
            duplicates = set(ids[ids.notna() & ids.duplicated(keep=False)])
            valid_rows = []
            source_rows = []
            missing_grades = []
            for row_idx, (_, row) in enumerate(df.iterrows()):
                try:
                    sid = ids.iloc[row_idx]
                    if pd.isna(sid):
                        raise ValueError('供应商编号不能为空或无效')
                    if sid in duplicates:
                        raise ValueError(f'供应商编号重复: {sid}（所有重复行均排除）')
                    row = row.copy()
                    row['供应商编号'] = sid
                    row = self._validate_numeric_row(row, allow_missing=True)
                    missing = [col for col in self._NUMERIC_COLUMNS if col in row and pd.isna(row[col])]
                    if missing:
                        print(f'警告：第{row_idx+2}行缺失数值将按有效行中位数或行业参考值填充: {missing}')
                    valid_rows.append(row)
                    source_rows.append(row_idx + 2)
                    missing_grades.append([col for col in self._GRADE_COLUMNS if pd.isna(row[col])])
                except (ValueError, TypeError) as exc:
                    print(f'警告：第{row_idx+2}行数据无效，已排除: {exc}')

            if not valid_rows:
                print('警告：没有有效的供应商数据')
                return []
            # Use only valid rows when estimating imputation or outlier thresholds.
            df = pd.DataFrame(valid_rows).reset_index(drop=True)
            
            try:
                from data_processor import DataProcessor
                dp = DataProcessor()
                df, _ = dp.process_dataframe(df)
            except ImportError:
                print("  注意: data_processor 模块未加载，跳过数据预处理")
            except Exception as e:
                print(f"  警告: 数据预处理失败 ({e})，继续使用原始数据")
            
            for row_idx, (idx, row) in enumerate(df.iterrows()):
                try:
                    row = row.copy()
                    for col in missing_grades[row_idx]:
                        if pd.notna(row[col]):
                            # A median may lie between two valid ordinal grades.
                            row[col] = round(float(row[col]))
                    row = self._validate_numeric_row(row, allow_missing=False)
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
                    print(f"警告：第{source_rows[row_idx]}行数据解析失败: {e}")
                    continue
            
            print(f"成功读取 {len(self.supplier_data)} 条供应商数据")
            
            return self.supplier_data
            
        except Exception as e:
            self.supplier_data = []
            print(f"读取Excel文件失败: {e}")
            return []

    @staticmethod
    def _normalize_supplier_id(value):
        if pd.isna(value):
            return None
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return None
            if value.is_integer():
                value = int(value)
        sid = str(value).strip()
        if not sid or sid.lower() in ('nan', 'none', 'null', 'inf', '-inf'):
            return None
        return sid

    def _validate_numeric_row(self, row, allow_missing):
        for col in self._NUMERIC_COLUMNS:
            if col not in row:
                continue
            value = row[col]
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                if allow_missing:
                    row[col] = np.nan
                    continue
                raise ValueError(f'{col} 缺失且未能填充')
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{col} 不能是布尔值')
            try:
                number = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{col} 必须是数值: {value}') from exc
            if not np.isfinite(number):
                raise ValueError(f'{col} 必须是有限数值: {value}')
            if col in self._GRADE_COLUMNS:
                if not number.is_integer() or not 1 <= number <= 5:
                    raise ValueError(f'{col} 必须是 1-5 的整数: {value}')
            elif col in self._PERCENT_COLUMNS:
                if not 0 <= number <= 100:
                    raise ValueError(f'{col} 必须在 0-100 范围内: {value}')
            elif number < 0:
                raise ValueError(f'{col} 不能为负数: {value}')
            row[col] = number
        return row
    
    def _parse_certifications(self, cert_string: str) -> List[str]:
        if pd.isna(cert_string) or cert_string == '':
            return []
        
        if isinstance(cert_string, str):
            return [cert.strip() for cert in cert_string.split(';') if cert.strip()]
        
        return [str(cert_string)]
    
    def convert_to_supplier_capability(self, excel_data: ExcelSupplierData) -> SupplierCapability:
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
        return [self.convert_to_supplier_capability(data) for data in self.supplier_data]
    
    def get_all_scoring_data(self) -> List[SupplierScoringData]:
        return [self.convert_to_supplier_scoring_data(data) for data in self.supplier_data]
    
    def generate_data_summary(self) -> Dict:
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
        summary = self.generate_data_summary()
        if not summary:
            print('没有可用的供应商数据，无法生成摘要')
            return
        
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
    template_data = {
        '供应商编号': ['S001', 'S002', 'S003', 'S004', 'S005'],
        '供应商名称': ['化工药剂有限公司', '环保科技股份', '水处理材料厂', '净化技术集团', '绿色化工公司'],
        '联系人': ['张经理', '李总监', '王厂长', '赵董事长', '刘主任'],
        '电话': ['13800138001', '13800138002', '13800138003', '13800138004', '13800138005'],
        '地址': ['北京市朝阳区', '上海市浦东新区', '广州市天河区', '深圳市南山区', '杭州市西湖区'],
        '药剂单价(元/吨)': [2800.0, 15000.0, 3200.0, 1800.0, 8500.0],
        '运输成本(元/吨)': [150.0, 200.0, 180.0, 120.0, 250.0],
        '付款周期(天)': [30.0, 45.0, 60.0, 15.0, 35.0],
        '药剂有效成分含量(%)': [95.0, 98.0, 92.0, 96.0, 94.0],
        '供货及时性(%)': [94.0, 98.0, 93.0, 97.0, 95.0],
        '库存能力(吨)': [5000.0, 3000.0, 6000.0, 8000.0, 4000.0],
        '交付能力(1-5级)': [4, 3, 4, 5, 4],
        '应急供应能力(1-5级)': [3, 2, 4, 5, 3],
        '安全管理水平(1-5级)': [4, 5, 4, 5, 4],
        '处理污泥效果达标率(%)': [88.0, 92.0, 85.0, 91.0, 87.0],
        '质量认证': ['ISO9001', 'ISO9001;ISO14001', 'ISO9001', 'ISO9001;ISO14001;OHSAS18001', 'ISO9001;ISO14001'],
        '环境认证': ['ISO14001', 'ISO14001;绿色认证', 'ISO14001', 'ISO14001;绿色认证;环保认证', 'ISO14001;绿色认证']
    }
    
    df = pd.DataFrame(template_data)
    df.to_excel(output_path, index=False, sheet_name='供应商数据')
    print(f"Excel模板文件已创建: {output_path}")
    return output_path
