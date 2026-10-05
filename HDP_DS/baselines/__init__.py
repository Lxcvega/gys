"""HDP-DS 基线对比模块

包含:
  bp_nsga2      — V2 Baseline (BP+NSGA-II 封装对比)
  legacy        — V1 原始代码 (完整BP+NSGA-II系统备份)
"""
from .bp_nsga2 import BPNSGA2Baseline

__all__ = ['BPNSGA2Baseline']
