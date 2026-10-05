"""
DataProcessor — 数据预处理管线

职责:
  1. 缺失值检测与填充（中位数 → 行业参考值兜底）
  2. 异常值检测与修正（IQR + MAD-Z 双重检测 → Winsorization 截尾）
  3. 可选的特征衍生与缺失标志列生成

设计原则:
  - 工作在 pandas DataFrame 层面，在 Excel 读取阶段执行
  - 下游所有模块（评分、NN、模糊、NSGA-II、博弈）自动获得清洗后数据
  - 默认不改变特征维度（10 维），不破坏已保存模型
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

try:
    from .data_model import SupplierCapability
except ImportError:
    SupplierCapability = None


# ─── 行业参考值（当同类中位数不可用时的兜底值） ───
INDUSTRY_REFERENCE: Dict[str, float] = {
    '药剂单价(元/吨)': 1600.0,
    '运输成本(元/吨)': 150.0,
    '付款周期(天)': 45.0,
    '药剂有效成分含量(%)': 92.0,
    '供货及时性(%)': 92.0,
    '库存能力(吨)': 5000.0,
    '交付能力(1-5级)': 4.0,
    '应急供应能力(1-5级)': 3.0,
    '安全管理水平(1-5级)': 4.0,
    '处理污泥效果达标率(%)': 90.0,
}

# 等级特征（1-5 级整数），跳过大尺度 IQR 截尾，只做边界保护
INTEGER_GRADE_FEATURES: List[str] = [
    '交付能力(1-5级)',
    '应急供应能力(1-5级)',
    '安全管理水平(1-5级)',
]

# 全部可处理的数值特征
NUMERIC_FEATURES: List[str] = list(INDUSTRY_REFERENCE.keys())


@dataclass
class DataProcessorConfig:
    """可配置参数"""
    impute_method: str = 'median'          # median | industry_ref
    outlier_method: str = 'iqr_mad'        # iqr_mad | none
    iqr_multiplier: float = 1.5            # IQR 倍数（越大越宽松）
    zscore_threshold: float = 3.5          # MAD-Z 阈值（越大越宽松）
    skip_grade_features: bool = True       # 等级特征跳过 IQR
    add_missing_indicators: bool = False   # 是否生成缺失标志列（默认关闭）
    add_derived_features: bool = False     # 是否生成衍生特征（默认关闭）
    verbose: bool = True                   # 是否打印处理报告


@dataclass
class ProcessingReport:
    """处理报告——记录每个步骤的影响"""
    n_rows: int = 0
    n_columns: int = 0
    missing_before: Dict[str, int] = field(default_factory=dict)
    missing_after: Dict[str, int] = field(default_factory=dict)
    n_values_imputed: int = 0
    outlier_log: List[Dict] = field(default_factory=list)
    n_outliers_corrected: int = 0
    grade_clamps: List[Dict] = field(default_factory=list)
    n_grade_clamped: int = 0

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"  数据预处理报告")
        print(f"{'=' * 60}")
        print(f"  数据集: {self.n_rows} 行 × {self.n_columns} 列")

        if self.missing_before:
            print(f"\n  [缺失值处理]:")
            for col, cnt in self.missing_before.items():
                if cnt > 0:
                    filled = self.missing_before.get(col, 0) - self.missing_after.get(col, 0)
                    print(f"    {col}: {cnt} 个缺失 -> 已填充 {filled} 个")

        if self.outlier_log:
            print(f"\n  [异常值修正] (IQR + MAD-Z):")
            for entry in self.outlier_log:
                print(f"    {entry['column']}: {entry['n_outliers']} 个异常值 "
                      f"(IQR={entry['iqr_outliers']}, Z={entry['z_outliers']}) "
                      f"-> 截尾到 [{entry['lower']:.1f}, {entry['upper']:.1f}]")

        if self.grade_clamps:
            print(f"\n  [等级特征边界保护]:")
            for entry in self.grade_clamps:
                print(f"    {entry['column']}: {entry['n_outliers']} 个越界值 -> 修正到 [1, 5]")

        print(f"\n  [完成] 共填充 {self.n_values_imputed} 个缺失值，"
              f"修正 {self.n_outliers_corrected + self.n_grade_clamped} 个异常/越界值")
        print(f"{'=' * 60}\n")


class DataProcessor:
    """
    数据预处理器

    用法:
        dp = DataProcessor()
        df_clean, report = dp.process_dataframe(df_raw)   # DataFrame 级
        caps_clean, report = dp.apply_to_capabilities(caps)  # SupplierCapability 级
    """

    def __init__(self, config: Optional[DataProcessorConfig] = None):
        self.config = config or DataProcessorConfig()

    # ──────────────────────────────────────────────
    #  步骤 1: 缺失值检测与填充
    # ──────────────────────────────────────────────

    def _detect_missing(self, df: pd.DataFrame) -> Dict[str, int]:
        """检测每列的缺失数量"""
        result = {}
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                n = int(df[col].isna().sum())
                if n > 0:
                    result[col] = n
        return result

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        缺失值填充:
          优先级: 同类中位数 → 行业参考值
          可选: 生成缺失标志列
        """
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue

            mask = df[col].isna()
            n_missing = mask.sum()
            if n_missing == 0:
                continue

            # 填充值: 中位数（抗异常）→ 行业参考值（兜底）
            if self.config.impute_method == 'median':
                fill_val = df[col].median()
                if pd.isna(fill_val):  # 整列缺失
                    fill_val = INDUSTRY_REFERENCE.get(col, 0.0)
            else:
                fill_val = INDUSTRY_REFERENCE.get(col, 0.0)

            df.loc[mask, col] = fill_val

            # 可选: 缺失标志列
            if self.config.add_missing_indicators:
                flag_col = f'{col}_is_missing'
                if flag_col not in df.columns:
                    df[flag_col] = 0
                df.loc[mask, flag_col] = 1

        return df

    # ──────────────────────────────────────────────
    #  步骤 2: 异常值检测与修正
    # ──────────────────────────────────────────────

    def _protect_grade_feature(self, df: pd.DataFrame, col: str) -> List[Dict]:
        """
        等级特征边界保护:
          - 不压缩有效范围（IQR 可能将 5 级误判为异常）
          - 只修复 <1 或 >5 的越界值
        """
        clamp_log = []

        n_low = int((df[col] < 1).sum())
        n_high = int((df[col] > 5).sum())
        n_total = n_low + n_high

        if n_total > 0:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            df.loc[df[col] < 1, col] = 1
            df.loc[df[col] > 5, col] = 5
            clamp_log.append({
                'column': col,
                'n_outliers': n_total,
                'range_before': f'[{col_min:.2f}, {col_max:.2f}]',
                'range_after': '[1, 5]',
            })

        return clamp_log

    def _correct_numeric_outliers(self, df: pd.DataFrame, col: str) -> Optional[Dict]:
        """
        数值特征的异常值修正:
          双重检测: IQR + MAD-Z（比纯 Z-Score 更鲁棒）
          修正方式: Winsorization 截尾（保留样本量）
        """
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        k = self.config.iqr_multiplier

        # IQR=0 时无法判断，跳过
        if IQR == 0:
            return None

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        # 检测方法 1: IQR
        iqr_outliers = (df[col] < lower) | (df[col] > upper)

        # 检测方法 2: MAD-Z（用 MAD 代替 std，免受异常值污染）
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        if mad > 0:
            modified_z = 0.6745 * (df[col] - median) / mad
            z_outliers = np.abs(modified_z) > self.config.zscore_threshold
        else:
            z_outliers = pd.Series([False] * len(df))

        # 双重检测 → 并集（宁抓勿放）
        combined = iqr_outliers | z_outliers
        n_outliers = int(combined.sum())

        if n_outliers == 0:
            return None

        # Winsorization: 截尾到边界
        df.loc[df[col] < lower, col] = lower
        df.loc[df[col] > upper, col] = upper

        return {
            'column': col,
            'n_outliers': n_outliers,
            'lower': round(lower, 2),
            'upper': round(upper, 2),
            'iqr_outliers': int(iqr_outliers.sum()),
            'z_outliers': int(z_outliers.sum()),
        }

    def correct_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """
        异常值修正入口:
          - 等级特征 → 边界保护（不压制有效区分度）
          - 数值特征 → IQR + MAD-Z 双重检测 → Winsorization
        """
        outlier_log: List[Dict] = []
        grade_log: List[Dict] = []

        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue

            if self.config.skip_grade_features and col in INTEGER_GRADE_FEATURES:
                # 等级特征: 只做边界保护
                clamps = self._protect_grade_feature(df, col)
                grade_log.extend(clamps)
            else:
                # 数值特征: IQR + MAD-Z
                result = self._correct_numeric_outliers(df, col)
                if result:
                    outlier_log.append(result)

        return df, outlier_log, grade_log

    # ──────────────────────────────────────────────
    #  可选步骤 3: 特征衍生
    # ──────────────────────────────────────────────

    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """可选: 生成衍生特征（默认关闭，开启后会改变特征维度）"""
        if not self.config.add_derived_features:
            return df

        # 成本效益比 = 单价 / 有效成分含量（越低越好）
        if all(c in df.columns for c in ['药剂单价(元/吨)', '药剂有效成分含量(%)']):
            df['成本效益比'] = df['药剂单价(元/吨)'] / df['药剂有效成分含量(%)'].clip(lower=1)

        return df

    # ──────────────────────────────────────────────
    #  主入口 1: DataFrame 级
    # ──────────────────────────────────────────────

    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ProcessingReport]:
        """
        完整预处理管线（DataFrame 级）

        参数:
            df: 原始 DataFrame（来自 pd.read_excel）

        返回:
            (清洗后的 DataFrame, 处理报告)
        """
        report = ProcessingReport(
            n_rows=len(df),
            n_columns=len(df.columns),
        )

        # 深拷贝避免修改原始数据
        df = df.copy()

        # ── 步骤 1: 缺失值 ──
        report.missing_before = self._detect_missing(df)
        df = self.impute_missing(df)
        report.missing_after = self._detect_missing(df)
        report.n_values_imputed = sum(
            report.missing_before.get(col, 0) - report.missing_after.get(col, 0)
            for col in report.missing_before
        )

        # ── 步骤 2: 异常值 ──
        df, outlier_log, grade_log = self.correct_outliers(df)
        report.outlier_log = outlier_log
        report.n_outliers_corrected = sum(e['n_outliers'] for e in outlier_log)
        report.grade_clamps = grade_log
        report.n_grade_clamped = sum(e['n_outliers'] for e in grade_log)

        # ── 步骤 3: 特征衍生（可选） ──
        df = self.derive_features(df)

        if self.config.verbose and (report.n_values_imputed > 0 or report.n_outliers_corrected > 0 or report.n_grade_clamped > 0):
            report.print_summary()
        elif self.config.verbose:
            print(f"\n  ✅ 数据无需清洗，所有数值均正常\n")

        return df, report

    # ──────────────────────────────────────────────
    #  主入口 2: SupplierCapability 列表级
    # ──────────────────────────────────────────────

    def apply_to_capabilities(self, capabilities: List) -> Tuple[List, ProcessingReport]:
        """
        直接处理 SupplierCapability 列表（原地修改）
        用于历史训练数据路径

        参数:
            capabilities: List[SupplierCapability]

        返回:
            (处理后的列表, 处理报告)
        """
        if not capabilities:
            return capabilities, ProcessingReport()

        records = []
        for s in capabilities:
            records.append({
                '药剂单价(元/吨)': s.material_unit_price,
                '运输成本(元/吨)': s.transport_cost,
                '付款周期(天)': s.payment_cycle,
                '药剂有效成分含量(%)': s.effective_ingredient_content,
                '供货及时性(%)': s.delivery_timeliness,
                '库存能力(吨)': s.inventory_capacity,
                '交付能力(1-5级)': s.delivery_capability,
                '应急供应能力(1-5级)': s.emergency_supply_capability,
                '安全管理水平(1-5级)': s.safety_management_level,
                '处理污泥效果达标率(%)': s.sludge_treatment_rate,
            })

        df = pd.DataFrame(records)
        df_clean, report = self.process_dataframe(df)

        for i, cap in enumerate(capabilities):
            cap.material_unit_price = float(df_clean.iloc[i]['药剂单价(元/吨)'])
            cap.transport_cost = float(df_clean.iloc[i]['运输成本(元/吨)'])
            cap.payment_cycle = float(df_clean.iloc[i]['付款周期(天)'])
            cap.effective_ingredient_content = float(df_clean.iloc[i]['药剂有效成分含量(%)'])
            cap.delivery_timeliness = float(df_clean.iloc[i]['供货及时性(%)'])
            cap.inventory_capacity = float(df_clean.iloc[i]['库存能力(吨)'])
            cap.delivery_capability = int(round(df_clean.iloc[i]['交付能力(1-5级)']))
            cap.emergency_supply_capability = int(round(df_clean.iloc[i]['应急供应能力(1-5级)']))
            cap.safety_management_level = int(round(df_clean.iloc[i]['安全管理水平(1-5级)']))
            cap.sludge_treatment_rate = float(df_clean.iloc[i]['处理污泥效果达标率(%)'])

        return capabilities, report
