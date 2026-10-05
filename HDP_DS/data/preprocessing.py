"""HDP-DS v3.1 时序预处理模块"""

import numpy as np
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataLeakageError(Exception):
    """数据泄漏异常"""
    pass


@dataclass
class ScalerManager:
    """管理动态数据标准化: 每个特征独立拟合，仅在训练集上拟合"""
    feature_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    fit_status: bool = False

    def fit(self, train_data: np.ndarray, feature_names: List[str]):
        """在训练数据上拟合 scaler"""
        self.feature_scalers = {}
        n_features = train_data.shape[-1]

        for i in range(n_features):
            col_data = train_data[..., i].flatten()
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) == 0:
                col_data = np.array([0.0])
            scaler = StandardScaler()
            scaler.fit(col_data.reshape(-1, 1))
            name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
            self.feature_scalers[name] = scaler

        self.fit_status = True

    def transform(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """应用标准化"""
        if not self.fit_status:
            raise ValueError("ScalerManager 尚未 fit")

        result = data.copy().astype(np.float32)
        n_features = data.shape[-1]

        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
            if name in self.feature_scalers:
                col = result[..., i].reshape(-1, 1)
                result[..., i] = self.feature_scalers[name].transform(col).reshape(result[..., i].shape)

        return result

    def inverse_transform(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """反标准化"""
        if not self.fit_status:
            return data

        result = data.copy()
        n_features = data.shape[-1]

        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
            if name in self.feature_scalers:
                col = result[..., i].reshape(-1, 1)
                result[..., i] = self.feature_scalers[name].inverse_transform(col).reshape(result[..., i].shape)

        return result


# ─── 时序交叉验证划分 ───────────────────────────────────────────

class GroupTimeSeriesSplit:
    """
    按供应商分组的时序交叉验证划分

    与 sklearn 的 TimeSeriesSplit 不同:
      - 按 supplier_id 分组
      - 每组内按时序划分
      - 保证同一供应商的数据不会跨训练/测试集泄露

    Usage:
        gts = GroupTimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in gts.split(X, groups=supplier_ids):
            ...
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Args:
            n_splits: 折数
            gap: 训练集与测试集之间的间隔 (防止时间泄露)
        """
        self.n_splits = n_splits
        self.gap = gap

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None, time_idx: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        生成 (train_index, test_index)

        Args:
            X: 特征数组
            groups: 供应商分组 ID 数组
            time_idx: 时间顺序索引 (升序)
        """
        if groups is None:
            # 无分组，直接用时序切分
            n = len(X)
            for i in range(self.n_splits):
                test_start = n - (self.n_splits - i) * (n // self.n_splits)
                train_end = test_start - self.gap
                if train_end > 0:
                    train_idx = np.arange(train_end)
                    test_idx = np.arange(test_start, n)
                    yield train_idx, test_idx
            return

        # 按供应商分组后划分
        unique_groups = np.unique(groups)

        # 为每个供应商获取其样本的时间排序索引
        from collections import defaultdict
        group_indices = defaultdict(list)
        for idx, g in enumerate(groups):
            group_indices[g].append(idx)

        # 对每个供应商内的样本按时间排序
        if time_idx is not None:
            for g in group_indices:
                indices = group_indices[g]
                indices.sort(key=lambda i: time_idx[i])
                group_indices[g] = indices

        # 按供应商划分 (留出部分供应商作为测试)
        all_groups = list(unique_groups)
        n_test_groups = max(1, len(all_groups) // (self.n_splits + 1))

        for i in range(self.n_splits):
            test_groups = set(all_groups[i * n_test_groups:(i + 1) * n_test_groups])
            train_idx, test_idx = [], []
            for g, indices in group_indices.items():
                if g in test_groups:
                    test_idx.extend(indices)
                else:
                    train_idx.extend(indices)

            yield np.array(train_idx), np.array(test_idx)


# ─── 时序预处理器 ───────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    scale_dynamic: bool = True            # 是否标准化动态数据
    scale_static: bool = True             # 是否标准化静态数据
    scale_method: str = 'standard'        # 'standard' | 'minmax'
    fill_missing: str = 'ffill'           # 'ffill' | 'median' | 'zero'
    clip_outliers: bool = True            # 是否裁剪异常值
    outlier_std_threshold: float = 4.0    # 异常值标准差阈值
    add_time_features: bool = True        # 是否添加时间特征 (月份正弦/余弦编码)
    verbose: bool = True


class TimeSeriesPreprocessor:
    """
    时序数据预处理器

    工作流程:
      1. 缺失值填充
      2. 异常值裁剪
      3. 标准化 (拟合训练集)
      4. 时间特征工程
      5. 分组时序划分
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.dynamic_scaler = ScalerManager()
        self.static_scaler: Optional[StandardScaler] = None
        self.static_minmax_scaler: Optional[MinMaxScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False

    # ═══════════════════════════════════════════════════════════════
    #  拟合与转换
    # ═══════════════════════════════════════════════════════════════

    def fit_transform(
        self,
        static_train: np.ndarray,      # [n_train, static_dim]
        dynamic_train: np.ndarray,     # [n_train, T, dynamic_dim]
        static_val: Optional[np.ndarray] = None,
        dynamic_val: Optional[np.ndarray] = None,
        static_test: Optional[np.ndarray] = None,
        dynamic_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple:
        """
        拟合预处理器并转换数据

        Returns:
            (static_train_scaled, dynamic_train_scaled,
             static_val_scaled, dynamic_val_scaled,
             static_test_scaled, dynamic_test_scaled)
        """
        if feature_names is None:
            feature_names = [
                'delivery_rate', 'quality_rate', 'complaint_count',
                'env_incident_count', 'cost_change_rate', 'service_response_time'
            ][:dynamic_train.shape[-1]]

        # 1. 动态数据标准化
        if self.config.scale_dynamic:
            self.dynamic_scaler.fit(dynamic_train, feature_names)
            dynamic_train_scaled = self.dynamic_scaler.transform(dynamic_train, feature_names)
            dynamic_val_scaled = self.dynamic_scaler.transform(dynamic_val, feature_names) \
                if dynamic_val is not None else None
            dynamic_test_scaled = self.dynamic_scaler.transform(dynamic_test, feature_names) \
                if dynamic_test is not None else None
        else:
            dynamic_train_scaled = dynamic_train.copy()
            dynamic_val_scaled = dynamic_val.copy() if dynamic_val is not None else None
            dynamic_test_scaled = dynamic_test.copy() if dynamic_test is not None else None

        # 2. 静态数据标准化
        if self.config.scale_static:
            if self.config.scale_method == 'standard':
                self.static_scaler = StandardScaler()
                static_train_scaled = self.static_scaler.fit_transform(static_train)
            else:
                self.static_minmax_scaler = MinMaxScaler()
                static_train_scaled = self.static_minmax_scaler.fit_transform(static_train)

            static_val_scaled = self._transform_static(static_val) if static_val is not None else None
            static_test_scaled = self._transform_static(static_test) if static_test is not None else None
        else:
            static_train_scaled = static_train.copy()
            static_val_scaled = static_val.copy() if static_val is not None else None
            static_test_scaled = static_test.copy() if static_test is not None else None

        self.is_fitted = True

        return (
            static_train_scaled, dynamic_train_scaled,
            static_val_scaled, dynamic_val_scaled,
            static_test_scaled, dynamic_test_scaled
        )

    def transform_only(
        self, static: np.ndarray, dynamic: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """仅转换 (用于新数据推断)"""
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，请先调用 fit_transform")

        if feature_names is None:
            feature_names = [
                'delivery_rate', 'quality_rate', 'complaint_count',
                'env_incident_count', 'cost_change_rate', 'service_response_time'
            ][:dynamic.shape[-1]]

        dynamic_scaled = self.dynamic_scaler.transform(dynamic, feature_names) \
            if self.config.scale_dynamic else dynamic.copy()
        static_scaled = self._transform_static(static) \
            if self.config.scale_static else static.copy()

        return static_scaled, dynamic_scaled

    def inverse_transform_dynamic(
        self, data: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """反标准化动态数据"""
        if feature_names is None:
            feature_names = [
                'delivery_rate', 'quality_rate', 'complaint_count',
                'env_incident_count', 'cost_change_rate', 'service_response_time'
            ][:data.shape[-1]]
        return self.dynamic_scaler.inverse_transform(data, feature_names)

    # ═══════════════════════════════════════════════════════════════
    #  静态辅助方法
    # ═══════════════════════════════════════════════════════════════

    def encode_categorical(self, values: List[str], column_name: str) -> np.ndarray:
        """标签编码分类变量"""
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
            return self.label_encoders[column_name].fit_transform(values)
        try:
            return self.label_encoders[column_name].transform(values)
        except Exception:
            # 处理未知类别
            le = self.label_encoders[column_name]
            result = np.full(len(values), -1, dtype=int)
            known_mask = np.isin(values, le.classes_)
            result[known_mask] = le.transform(np.array(values)[known_mask])
            return result

    def add_time_features(self, timestamps: List[str]) -> np.ndarray:
        """
        添加时间特征: 月份正弦/余弦编码

        Args:
            timestamps: ["2024-01", "2024-02", ...] 格式

        Returns:
            [len(timestamps), 2]  (sin, cos)
        """
        months = []
        for ts in timestamps:
            try:
                month = int(ts.split('-')[1])
            except Exception:
                month = 1
            months.append(month)

        months = np.array(months, dtype=np.float32)
        angle = 2 * np.pi * (months - 1) / 12.0
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def clip_outliers(self, data: np.ndarray, feature_axis: int = -1) -> np.ndarray:
        """
        裁剪异常值 (基于标准差)

        对每个特征独立计算 mean ± threshold * std
        """
        if not self.config.clip_outliers:
            return data

        result = data.copy()
        threshold = self.config.outlier_std_threshold
        n_features = data.shape[feature_axis]

        for i in range(n_features):
            col = result[..., i]
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std == 0:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
            result[..., i] = np.clip(col, lower, upper)

        return result

    def fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        result = data.copy()
        method = self.config.fill_missing

        if method == 'ffill':
            # 前向填充 (沿时间轴)
            for i in range(result.shape[-1]):
                col = result[..., i]
                mask = np.isnan(col)
                if mask.any():
                    col_pd = pd.Series(col.flatten())
                    col_pd = col_pd.ffill().bfill()
                    result[..., i] = col_pd.values.reshape(col.shape)
        elif method == 'median':
            for i in range(result.shape[-1]):
                col = result[..., i]
                mask = np.isnan(col)
                if mask.any():
                    median = np.nanmedian(col)
                    col[mask] = median
                    result[..., i] = col
        elif method == 'zero':
            for i in range(result.shape[-1]):
                col = result[..., i]
                mask = np.isnan(col)
                if mask.any():
                    col[mask] = 0.0
                    result[..., i] = col

        return result

    def validate_no_leakage(
        self, train_ids: List[str], test_ids: List[str]
    ) -> bool:
        """
        验证无数据泄漏

        检查 train 和 test 中没有相同的 supplier_id
        """
        train_set = set(train_ids)
        test_set = set(test_ids)
        overlap = train_set & test_set
        if overlap:
            raise DataLeakageError(
                f"数据泄漏检测: {len(overlap)} 个供应商同时出现在训练集和测试集: {list(overlap)[:5]}"
            )
        return True

    def _transform_static(self, static: np.ndarray) -> np.ndarray:
        """转换静态数据"""
        if self.static_scaler is not None and self.config.scale_method == 'standard':
            return self.static_scaler.transform(static)
        elif self.static_minmax_scaler is not None:
            return self.static_minmax_scaler.transform(static)
        return static.copy()


# ─── 便捷函数 ────────────────────────────────────────────────────

def create_rolling_windows(
    data: np.ndarray,
    window_size: int,
    forecast_horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从单条时间序列创建滚动窗口

    Args:
        data: [T, N] 时间序列
        window_size: 窗口长度
        forecast_horizon: 预测步长
        stride: 步长

    Returns:
        (X_windows, y_targets)
        X_windows: [n_windows, window_size, N]
        y_targets: [n_windows, N] (取预测期后的值)
    """
    T = data.shape[0]
    X_windows, y_targets = [], []

    for t in range(0, T - window_size - forecast_horizon + 1, stride):
        X_windows.append(data[t:t + window_size])
        y_targets.append(data[t + window_size + forecast_horizon - 1])

    if len(X_windows) == 0:
        return np.array([]).reshape(0, window_size, data.shape[1]), \
               np.array([]).reshape(0, data.shape[1])

    return np.array(X_windows), np.array(y_targets)
