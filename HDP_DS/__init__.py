"""HDP-DS v3.1 — Multi-source Predictive Fusion 异构数据预测驱动的动态供应商选择框架"""

__version__ = "3.1.0"

# 数据模块
from HDP_DS.data.data_loader import HDPDataLoader, HDPDataset
from HDP_DS.data.preprocessing import TimeSeriesPreprocessor, GroupTimeSeriesSplit
from HDP_DS.data.generation.mock_data import generate_mock_data

# 模型模块
from HDP_DS.models.static_encoder import XGBoostStaticEncoder, StaticEncoderConfig
from HDP_DS.models.tft_encoder import TFTDynamicEncoder, TFTConfig
from HDP_DS.models.fusion import CrossAttentionFusion, FusionConfig
from HDP_DS.models.multitask_head import MultiTaskPredictionHead, MultiTaskConfig
from HDP_DS.models.hdp_model import HDPModel, HDPModelConfig

# 优化模块
from HDP_DS.optimization.nsga2 import HDPNSGA2, NSGA2Config
from HDP_DS.optimization.decision import TOPSISDecision, TOPSISConfig

# 训练模块
from HDP_DS.training.loss import MultiTaskLoss
from HDP_DS.training.train import Trainer, TrainerConfig

# 评估模块
from HDP_DS.evaluation.metrics import EvaluationMetrics, ExperimentComparator
from HDP_DS.evaluation.ablation import AblationConfig, AblationRunner

# 反馈模块
from HDP_DS.feedback.feedback_loop import FeedbackLoop, FeedbackConfig

# 基线模块
from HDP_DS.baselines.bp_nsga2 import BPNSGA2Baseline

__all__ = [
    '__version__',
    'HDPDataLoader', 'HDPDataset', 'TimeSeriesPreprocessor',
    'GroupTimeSeriesSplit', 'generate_mock_data',
    'XGBoostStaticEncoder', 'StaticEncoderConfig',
    'TFTDynamicEncoder', 'TFTConfig',
    'CrossAttentionFusion', 'FusionConfig',
    'MultiTaskPredictionHead', 'MultiTaskConfig',
    'HDPModel', 'HDPModelConfig',
    'HDPNSGA2', 'NSGA2Config',
    'TOPSISDecision', 'TOPSISConfig',
    'MultiTaskLoss', 'Trainer', 'TrainerConfig',
    'EvaluationMetrics', 'ExperimentComparator',
    'AblationConfig', 'AblationRunner',
    'FeedbackLoop', 'FeedbackConfig',
    'BPNSGA2Baseline',
]
