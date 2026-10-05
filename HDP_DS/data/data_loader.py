"""HDP-DS v3.1 数据加载器: 从 Excel 加载静态特征表和动态时序表"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StaticFeatures:
    """供应商静态特征 (6维)"""
    enterprise_scale: float
    registered_capital: float
    iso_certified: int
    env_certified: int
    credit_rating: int
    enterprise_age: float

    def to_array(self) -> np.ndarray:
        return np.array([self.enterprise_scale, self.registered_capital,
                         self.iso_certified, self.env_certified,
                         self.credit_rating, self.enterprise_age], dtype=np.float32)


@dataclass
class DynamicSequence:
    """供应商动态时间序列 (月度)"""
    supplier_id: str
    timestamps: List[str]
    values: np.ndarray  # [T, 6]
    feature_names: List[str] = field(default_factory=lambda: [
        'delivery_rate', 'quality_rate', 'complaint_count',
        'env_incident_count', 'cost_change_rate', 'service_response_time'
    ])

    @property
    def length(self) -> int:
        return self.values.shape[0]

    @property
    def num_features(self) -> int:
        return self.values.shape[1]


@dataclass
class KPI_Targets:
    """未来KPI目标向量 (5维)"""
    delivery_reliability: float    # 交付可靠性
    quality_stability: float       # 质量稳定性
    environmental_risk: float      # 环境风险 (越低越好)
    cost_stability: float          # 成本稳定性
    service_capability: float      # 服务能力

    def to_array(self) -> np.ndarray:
        return np.array([
            self.delivery_reliability,
            self.quality_stability,
            self.environmental_risk,
            self.cost_stability,
            self.service_capability
        ], dtype=np.float32)


@dataclass
class RollingSample:
    """一个滚动窗口样本"""
    supplier_id: str
    static_features: StaticFeatures
    dynamic_seq: np.ndarray       # shape [window_size, 6]
    target_kpi: np.ndarray        # shape [5]
    timestamp: str                # 预测时间点


@dataclass
class HDPDataset:
    """完整数据集"""
    samples: List[RollingSample]
    supplier_ids: List[str]
    static_dim: int = 6
    dynamic_dim: int = 6
    kpi_dim: int = 5
    window_size: int = 12         # 默认12个月

    def __len__(self) -> int:
        return len(self.samples)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """转换为 (static, dynamic, target) 三元组"""
        static_list = [s.static_features.to_array() for s in self.samples]
        dynamic_list = [s.dynamic_seq for s in self.samples]
        target_list = [s.target_kpi for s in self.samples]
        return (
            np.array(static_list, dtype=np.float32),
            np.array(dynamic_list, dtype=np.float32),
            np.array(target_list, dtype=np.float32)
        )


# ─── 数据加载器 ─────────────────────────────────────────────────

class HDPDataLoader:
    """
    HDP-DS 数据加载器

    从 Excel 文件中读取:
      - 静态特征表 (Sheet: "静态数据")
      - 动态时序表 (Sheet: "动态数据")
      - KPI目标表  (Sheet: "KPI标签")

    支持 Rolling Window 构造样本。
    """

    def __init__(
        self,
        static_path: Optional[str] = None,
        dynamic_path: Optional[str] = None,
        window_size: int = 12,
        forecast_horizon: int = 3,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42
    ):
        """
        Args:
            static_path: 静态特征 Excel 路径
            dynamic_path: 动态时序 Excel 路径 (可与 static 同文件不同 sheet)
            window_size: 滚动窗口大小 (月)
            forecast_horizon: 预测步长 (月)
            val_split: 验证集比例
            test_split: 测试集比例
        """
        self.static_path = static_path
        self.dynamic_path = dynamic_path or static_path
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state

        # 存储原始数据
        self.static_df: Optional[pd.DataFrame] = None
        self.dynamic_df: Optional[pd.DataFrame] = None
        self.kpi_df: Optional[pd.DataFrame] = None

        # 数据信息
        self.supplier_info: Dict[str, Dict] = {}

    # ═══════════════════════════════════════════════════════════════
    #  公开接口
    # ═══════════════════════════════════════════════════════════════

    def load_from_excel(self, excel_path: str) -> bool:
        """
        从单 Excel 文件加载所有数据 (多 Sheet)

        Sheet 约定:
          - "静态数据"   ->  static_features
          - "动态数据"   ->  dynamic_sequences
          - "KPI标签"    ->  kpi_targets
        """
        if not os.path.exists(excel_path):
            print(f"错误: 文件不存在 {excel_path}")
            return False

        try:
            xls = pd.ExcelFile(excel_path)

            # 加载静态数据
            if '静态数据' in xls.sheet_names:
                self.static_df = pd.read_excel(excel_path, sheet_name='静态数据')
                print(f"  [加载] 静态数据: {len(self.static_df)} 行")
            else:
                print("  警告: 未找到 '静态数据' Sheet，尝试使用默认列")
                self.static_df = self._create_default_static()

            # 加载动态数据
            if '动态数据' in xls.sheet_names:
                self.dynamic_df = pd.read_excel(excel_path, sheet_name='动态数据')
                print(f"  [加载] 动态数据: {len(self.dynamic_df)} 行")
            else:
                print("  错误: 未找到 '动态数据' Sheet")
                return False

            # 加载 KPI 标签
            if 'KPI标签' in xls.sheet_names:
                self.kpi_df = pd.read_excel(excel_path, sheet_name='KPI标签')
                print(f"  [加载] KPI标签: {len(self.kpi_df)} 行")
            else:
                print("  警告: 未找到 'KPI标签' Sheet，将使用动态数据最后几期生成")

            self._build_supplier_info()
            return True

        except Exception as e:
            print(f"错误: 加载 Excel 失败: {e}")
            return False

    def build_dataset(self) -> HDPDataset:
        """
        构建滚动窗口数据集

        返回 HDPDataset，包含所有供应商的 RollingSample。
        禁止随机打乱 — 保持时序顺序，避免数据泄漏。
        """
        if self.dynamic_df is None:
            raise ValueError("请先调用 load_from_excel()")

        samples: List[RollingSample] = []
        supplier_ids = self.dynamic_df['supplier_id'].unique() if 'supplier_id' in self.dynamic_df.columns \
            else self.dynamic_df.iloc[:, 0].unique()

        for sid in supplier_ids:
            # 按时间排序
            dyn = self.dynamic_df[self.dynamic_df['supplier_id'] == sid] \
                if 'supplier_id' in self.dynamic_df.columns \
                else self.dynamic_df[self.dynamic_df.iloc[:, 0] == sid]
            dyn = dyn.sort_values('period') if 'period' in dyn.columns else dyn.sort_index()

            # 解析为数值矩阵
            dyn_values = self._parse_dynamic_values(dyn)
            timestamps = self._parse_timestamps(dyn)

            # 获取静态特征
            static = self._get_static_features(sid)

            # 获取 KPI 标签
            kpi_values = self._get_kpi_targets(sid)

            if dyn_values.shape[0] < self.window_size + self.forecast_horizon:
                print(f"  跳过 {sid}: 时序长度 {dyn_values.shape[0]} < 窗口{self.window_size}+预测{self.forecast_horizon}")
                continue

            # 滑动窗口构造样本
            max_start = dyn_values.shape[0] - self.window_size - self.forecast_horizon + 1
            for t in range(max_start):
                seq = dyn_values[t:t + self.window_size]   # [window, 6]
                # 目标 = 预测期之后的 KPI 值 (或动态特征的聚合)
                if kpi_values is not None and t < len(kpi_values):
                    target = kpi_values[min(t, len(kpi_values) - 1)]
                else:
                    target = self._compute_kpi_from_seq(
                        dyn_values[t + self.window_size:t + self.window_size + self.forecast_horizon]
                    )

                sample = RollingSample(
                    supplier_id=str(sid),
                    static_features=static,
                    dynamic_seq=seq,
                    target_kpi=target,
                    timestamp=timestamps[t + self.window_size - 1] if t + self.window_size - 1 < len(timestamps) else ""
                )
                samples.append(sample)

        dataset = HDPDataset(
            samples=samples,
            supplier_ids=[str(s) for s in supplier_ids],
            window_size=self.window_size
        )

        print(f"\n  [数据集] 总样本数: {len(samples)}, 供应商数: {len(supplier_ids)}")
        return dataset

    def train_val_test_split(
        self, dataset: HDPDataset
    ) -> Tuple[HDPDataset, HDPDataset, HDPDataset]:
        """
        时序感知的 train/val/test 划分

        按供应商分组，每个供应商的时间序列按时间戳切分。
        禁止随机打乱。
        """
        # 按供应商分组
        from collections import defaultdict
        supplier_samples: Dict[str, List[RollingSample]] = defaultdict(list)
        for s in dataset.samples:
            supplier_samples[s.supplier_id].append(s)

        train_samples, val_samples, test_samples = [], [], []

        for sid, samples in supplier_samples.items():
            samples.sort(key=lambda x: x.timestamp)  # 按时间排序
            n = len(samples)
            n_test = max(1, int(n * self.test_split))
            n_val = max(1, int(n * self.val_split))
            n_train = n - n_test - n_val

            if n_train <= 0:
                n_train = max(1, n // 2)
                n_val = max(1, (n - n_train) // 2)
                n_test = n - n_train - n_val

            # 时序切分 (不 shuffle)
            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train + n_val])
            test_samples.extend(samples[n_train + n_val:])

        train_ds = HDPDataset(train_samples, dataset.supplier_ids,
                              window_size=dataset.window_size)
        val_ds = HDPDataset(val_samples, dataset.supplier_ids,
                            window_size=dataset.window_size)
        test_ds = HDPDataset(test_samples, dataset.supplier_ids,
                             window_size=dataset.window_size)

        print(f"  [划分] 训练: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")
        return train_ds, val_ds, test_ds

    # ═══════════════════════════════════════════════════════════════
    #  内部方法
    # ═══════════════════════════════════════════════════════════════

    def _parse_dynamic_values(self, dyn_df: pd.DataFrame) -> np.ndarray:
        """从 DataFrame 解析动态数值矩阵 [T, 6]"""
        # 动态特征列名映射
        dyn_cols = [
            'delivery_rate', 'quality_rate', 'complaint_count',
            'env_incident_count', 'cost_change_rate', 'service_response_time'
        ]
        available = [c for c in dyn_cols if c in dyn_df.columns]
        if not available:
            # 尝试使用第2-7列
            numeric_cols = dyn_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 6:
                available = numeric_cols[:6]
            else:
                raise ValueError(f"动态数据缺少数值列, 可用: {dyn_df.columns.tolist()}")

        values = dyn_df[available].values.astype(np.float32)

        # 缺失值填充: 前向填充 + 后向填充
        df_temp = pd.DataFrame(values)
        df_temp = df_temp.ffill().bfill()
        values = df_temp.values.astype(np.float32)

        return values

    def _parse_timestamps(self, dyn_df: pd.DataFrame) -> List[str]:
        """解析时间戳"""
        if 'period' in dyn_df.columns:
            return dyn_df['period'].astype(str).tolist()
        return [str(i) for i in range(len(dyn_df))]

    def _get_static_features(self, supplier_id: str) -> StaticFeatures:
        """获取供应商静态特征"""
        if self.static_df is None:
            # 默认值
            return StaticFeatures(
                enterprise_scale=0.5, registered_capital=1000.0,
                iso_certified=1, env_certified=0,
                credit_rating=2, enterprise_age=10.0
            )

        row = self.static_df[self.static_df['supplier_id'] == supplier_id] \
            if 'supplier_id' in self.static_df.columns \
            else self.static_df.iloc[[0]]

        if len(row) == 0:
            return StaticFeatures(
                enterprise_scale=0.5, registered_capital=1000.0,
                iso_certified=1, env_certified=0,
                credit_rating=2, enterprise_age=10.0
            )

        row = row.iloc[0]
        return StaticFeatures(
            enterprise_scale=float(row.get('enterprise_scale', 0.5)),
            registered_capital=float(row.get('registered_capital', 1000.0)),
            iso_certified=int(row.get('iso_certified', 0)),
            env_certified=int(row.get('env_certified', 0)),
            credit_rating=int(row.get('credit_rating', 2)),
            enterprise_age=float(row.get('enterprise_age', 10.0))
        )

    def _get_kpi_targets(self, supplier_id: str) -> Optional[np.ndarray]:
        """获取供应商 KPI 目标向量序列 [T, 5]"""
        if self.kpi_df is None:
            return None

        kpi = self.kpi_df[self.kpi_df['supplier_id'] == supplier_id] \
            if 'supplier_id' in self.kpi_df.columns \
            else self.kpi_df

        if len(kpi) == 0:
            return None

        kpi_cols = [
            'delivery_reliability', 'quality_stability',
            'environmental_risk', 'cost_stability', 'service_capability'
        ]
        available = [c for c in kpi_cols if c in kpi.columns]
        if len(available) < 5:
            return None

        values = kpi[available].values.astype(np.float32)
        return values

    def _compute_kpi_from_seq(self, future_seq: np.ndarray) -> np.ndarray:
        """
        从未来动态序列中计算 KPI 向量 [5]

        当没有显式 KPI 标签时使用此方法。
        """
        if len(future_seq) == 0:
            return np.zeros(5, dtype=np.float32)

        mean = future_seq.mean(axis=0)  # [6]
        # 转换为 5 维 KPI
        p1 = mean[0] / 100.0                    # 交付率 → 交付可靠性
        p2 = mean[1] / 100.0                    # 质量率 → 质量稳定性
        p3 = 1.0 - min(1.0, mean[3] / 10.0)     # 环保事件 → 环境风险(逆)
        p4 = 1.0 - min(1.0, abs(mean[4]) / 20.0)  # 成本变化 → 成本稳定性(逆)
        p5 = 1.0 - min(1.0, mean[5] / 48.0)     # 响应时间 → 服务能力(逆)
        return np.array([p1, p2, p3, p4, p5], dtype=np.float32)

    def _build_supplier_info(self):
        """构建供应商信息索引"""
        if self.static_df is not None:
            for _, row in self.static_df.iterrows():
                sid = str(row.get('supplier_id', ''))
                self.supplier_info[sid] = {
                    'name': row.get('supplier_name', f'供应商{sid}'),
                }

    def _create_default_static(self) -> pd.DataFrame:
        """创建默认静态数据 (当 Excel 中没有静态 Sheet 时)"""
        if self.dynamic_df is not None:
            supplier_ids = self.dynamic_df['supplier_id'].unique() \
                if 'supplier_id' in self.dynamic_df.columns \
                else self.dynamic_df.iloc[:, 0].unique()
            data = []
            for i, sid in enumerate(supplier_ids):
                data.append({
                    'supplier_id': sid,
                    'enterprise_scale': 0.3 + 0.7 * (i / max(1, len(supplier_ids) - 1)),
                    'registered_capital': 500 + 5000 * (i / max(1, len(supplier_ids) - 1)),
                    'iso_certified': 1 if i % 3 != 0 else 0,
                    'env_certified': 1 if i % 4 != 0 else 0,
                    'credit_rating': min(3, i % 4),
                    'enterprise_age': 5 + 25 * (i / max(1, len(supplier_ids) - 1)),
                })
            return pd.DataFrame(data)
        return pd.DataFrame()

    def print_data_summary(self):
        """打印数据摘要"""
        print("\n" + "=" * 55)
        print("  HDP-DS 数据摘要")
        print("=" * 55)
        if self.static_df is not None:
            print(f"  静态特征: {len(self.static_df)} 家供应商, "
                  f"{len(self.static_df.columns)} 列")
        if self.dynamic_df is not None:
            print(f"  动态时序: {len(self.dynamic_df)} 行, "
                  f"{self.dynamic_df['supplier_id'].nunique() if 'supplier_id' in self.dynamic_df.columns else '?'} 家供应商")
        if self.kpi_df is not None:
            print(f"  KPI标签:   {len(self.kpi_df)} 行")
        print(f"  窗口大小:  {self.window_size} 月")
        print(f"  预测步长:  {self.forecast_horizon} 月")
        print("=" * 55)
