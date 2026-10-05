"""HDP-DS v3.1 消融实验模块"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import copy


@dataclass
class AblationConfig:
    """消融实验配置"""
    use_static: bool = True
    use_dynamic: bool = True
    use_attention: bool = True
    use_multitask: bool = True
    use_feedback: bool = True
    experiment_name: str = "HDP-DS Full"
    description: str = "完整 HDP-DS v3.1 模型"

    @staticmethod
    def get_default_experiments() -> Dict[str, 'AblationConfig']:
        """获取默认消融实验组"""
        full = AblationConfig(experiment_name="HDP-DS Full", description="完整 HDP-DS v3.1")
        no_static = AblationConfig(
            use_static=False, use_attention=False, experiment_name="HDP-DS w/o Static", description="去除静态编码器"
        )
        no_dynamic = AblationConfig(
            use_dynamic=False, use_attention=False, experiment_name="HDP-DS w/o Dynamic", description="去除动态编码器"
        )
        no_attention = AblationConfig(
            use_attention=False, experiment_name="HDP-DS w/o Attention", description="去除 Cross Attention"
        )
        no_multitask = AblationConfig(
            use_multitask=False, experiment_name="HDP-DS w/o Multi-task", description="去除多任务"
        )
        no_feedback = AblationConfig(
            use_static=True, use_dynamic=True,
            use_attention=True, use_multitask=True, use_feedback=False,
            experiment_name="HDP-DS w/o Feedback",
            description="去除闭环反馈"
        )

        return {
            'full': full,
            'no_static': no_static,
            'no_dynamic': no_dynamic,
            'no_attention': no_attention,
            'no_multitask': no_multitask,
            'no_feedback': no_feedback,
        }

    def get_active_components(self) -> List[str]:
        """返回当前启用的组件列表"""
        components = []
        if self.use_static:
            components.append('Static Encoder')
        if self.use_dynamic:
            components.append('TFT Encoder')
        if self.use_attention:
            components.append('Cross Attention')
        if self.use_multitask:
            components.append('Multi-task Head')
        if self.use_feedback:
            components.append('Feedback Loop')
        return components

    def print_config(self):
        """打印当前配置"""
        print(f"\n  [消融] {self.experiment_name}")
        print(f"  {'─' * 40}")
        print(f"  use_static:    {self.use_static}")
        print(f"  use_dynamic:   {self.use_dynamic}")
        print(f"  use_attention: {self.use_attention}")
        print(f"  use_multitask: {self.use_multitask}")
        print(f"  use_feedback:  {self.use_feedback}")
        print(f"  {'─' * 40}")
        print(f"  激活组件: {', '.join(self.get_active_components())}")


class AblationRunner:
    """
    消融实验运行器

    按消融配置运行实验，收集并对比结果。
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def run_experiment(
        self,
        config: AblationConfig,
        train_func: callable,
        eval_func: callable
    ) -> Dict[str, float]:
        """
        运行单个消融实验

        Args:
            config: 消融配置
            train_func: 训练函数 (接收 config 参数)
            eval_func: 评估函数

        Returns:
            metrics: {metric_name: value}
        """
        print(f"\n{'=' * 55}")
        print(f"  运行消融实验: {config.experiment_name}")
        print(f"{'=' * 55}")
        config.print_config()

        # 训练
        model = train_func(config)

        # 评估
        metrics = eval_func(model)

        self.results[config.experiment_name] = metrics
        return metrics

    def run_all_ablations(
        self,
        experiments: Dict[str, AblationConfig],
        train_func: callable,
        eval_func: callable
    ) -> Dict[str, Dict[str, float]]:
        """
        运行所有消融实验

        Args:
            experiments: {实验名: 配置}
            train_func: 训练函数
            eval_func: 评估函数

        Returns:
            {实验名: {指标: 值}}
        """
        for name, config in experiments.items():
            self.run_experiment(config, train_func, eval_func)

        self.print_ablation_table()
        return self.results

    def print_ablation_table(self):
        """打印消融实验对比表"""
        if not self.results:
            print("  无消融结果")
            return

        print(f"\n{'=' * 80}")
        print(f"  消融实验对比表")
        print(f"{'=' * 80}")

        # 获取所有指标名
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        # 表头
        header = f"  {'实验':25s}" + "".join([f"{m:>12s}" for m in all_metrics])
        print(f"  {'─' * 80}")
        print(header)
        print(f"  {'─' * 80}")

        for exp_name, metrics in self.results.items():
            row = f"  {exp_name:25s}"
            for m in all_metrics:
                val = metrics.get(m, float('nan'))
                row += f"{val:>12.4f}"
            print(row)

        print(f"  {'─' * 80}")

    def get_ablation_summary(self) -> Dict[str, Any]:
        """获取消融摘要"""
        return {
            'n_experiments': len(self.results),
            'results': self.results
        }
