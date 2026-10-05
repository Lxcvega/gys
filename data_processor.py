import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


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

INTEGER_GRADE_FEATURES: List[str] = [
    '交付能力(1-5级)',
    '应急供应能力(1-5级)',
    '安全管理水平(1-5级)',
]

NUMERIC_FEATURES: List[str] = list(INDUSTRY_REFERENCE.keys())


@dataclass
class DataProcessorConfig:
    impute_method: str = 'median'
    outlier_method: str = 'iqr_mad'
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.5
    skip_grade_features: bool = True
    add_missing_indicators: bool = False
    add_derived_features: bool = False
    verbose: bool = True


@dataclass
class ProcessingReport:
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

    def __init__(self, config: Optional[DataProcessorConfig] = None):
        self.config = config or DataProcessorConfig()

    def _detect_missing(self, df: pd.DataFrame) -> Dict[str, int]:
        result = {}
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                n = int(df[col].isna().sum())
                if n > 0:
                    result[col] = n
        return result

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue

            mask = df[col].isna()
            n_missing = mask.sum()
            if n_missing == 0:
                continue

            if self.config.impute_method == 'median':
                fill_val = df[col].median()
                if pd.isna(fill_val):
                    fill_val = INDUSTRY_REFERENCE.get(col, 0.0)
            else:
                fill_val = INDUSTRY_REFERENCE.get(col, 0.0)

            df.loc[mask, col] = fill_val

            if self.config.add_missing_indicators:
                flag_col = f'{col}_is_missing'
                if flag_col not in df.columns:
                    df[flag_col] = 0
                df.loc[mask, flag_col] = 1

        return df

    def _protect_grade_feature(self, df: pd.DataFrame, col: str) -> List[Dict]:
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
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        k = self.config.iqr_multiplier

        if IQR == 0:
            return None

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        iqr_outliers = (df[col] < lower) | (df[col] > upper)

        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        if mad > 0:
            modified_z = 0.6745 * (df[col] - median) / mad
            z_outliers = np.abs(modified_z) > self.config.zscore_threshold
        else:
            z_outliers = pd.Series([False] * len(df))

        combined = iqr_outliers | z_outliers
        n_outliers = int(combined.sum())

        if n_outliers == 0:
            return None

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
        outlier_log: List[Dict] = []
        grade_log: List[Dict] = []

        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue

            if self.config.skip_grade_features and col in INTEGER_GRADE_FEATURES:
                clamps = self._protect_grade_feature(df, col)
                grade_log.extend(clamps)
            else:
                result = self._correct_numeric_outliers(df, col)
                if result:
                    outlier_log.append(result)

        return df, outlier_log, grade_log

    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.add_derived_features:
            return df

        if all(c in df.columns for c in ['药剂单价(元/吨)', '药剂有效成分含量(%)']):
            df['成本效益比'] = df['药剂单价(元/吨)'] / df['药剂有效成分含量(%)'].clip(lower=1)

        return df

    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ProcessingReport]:
        report = ProcessingReport(
            n_rows=len(df),
            n_columns=len(df.columns),
        )

        df = df.copy()

        report.missing_before = self._detect_missing(df)
        df = self.impute_missing(df)
        report.missing_after = self._detect_missing(df)
        report.n_values_imputed = sum(
            report.missing_before.get(col, 0) - report.missing_after.get(col, 0)
            for col in report.missing_before
        )

        df, outlier_log, grade_log = self.correct_outliers(df)
        report.outlier_log = outlier_log
        report.n_outliers_corrected = sum(e['n_outliers'] for e in outlier_log)
        report.grade_clamps = grade_log
        report.n_grade_clamped = sum(e['n_outliers'] for e in grade_log)

        df = self.derive_features(df)

        if self.config.verbose and (report.n_values_imputed > 0 or report.n_outliers_corrected > 0 or report.n_grade_clamped > 0):
            report.print_summary()
        elif self.config.verbose:
            print(f"\n  ✅ 数据无需清洗，所有数值均正常\n")

        return df, report

    def apply_to_capabilities(self, capabilities: List) -> Tuple[List, ProcessingReport]:
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