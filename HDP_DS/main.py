"""HDP-DS v3.1 主入口"""

import argparse
import numpy as np
import os

from HDP_DS import (
    HDPDataLoader, HDPDataset,
    HDPModel, HDPModelConfig,
    HDPNSGA2, NSGA2Config,
    TOPSISDecision, TOPSISConfig,
    EvaluationMetrics, ExperimentComparator,
    AblationConfig, AblationRunner,
    FeedbackLoop, FeedbackConfig,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
DEFAULT_EXCEL_PATH = os.path.join(RAW_DIR, '供应商数据.xlsx')
DEFAULT_HISTORICAL_PATH = os.path.join(RAW_DIR, '历史训练数据_模板.xlsx')


def _load_dataset(excel_path, window_size):
    """Fail explicitly on unusable real data; simulations require --mode demo."""
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f'数据文件不存在: {excel_path}；模拟演示请指定 --mode demo')
    loader = HDPDataLoader(window_size=window_size)
    if not loader.load_from_excel(excel_path):
        raise ValueError(f'数据加载失败: {excel_path}；请检查 Excel 工作表和数据格式')
    try:
        dataset = loader.build_dataset()
        if len(dataset) == 0:
            raise ValueError('没有可用的滚动窗口样本，请检查时序长度和 --window')
        arrays = dataset.to_numpy()
        expected_shapes = ((len(dataset), 6), (len(dataset), window_size, 6), (len(dataset), 5))
        for name, values, shape in zip(('静态特征', '动态特征', 'KPI标签'), arrays, expected_shapes):
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f'{name}的形状或数值无效，预期 {shape}，实际 {values.shape}')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f'数据集无效: {excel_path}: {exc}') from exc
    return loader, dataset


def run_full_workflow(
    excel_path: str = DEFAULT_EXCEL_PATH,
    historical_path: str = DEFAULT_HISTORICAL_PATH,
    window_size: int = 12,
    n_epochs: int = 100
):
    """运行完整 HDP-DS 工作流: 加载数据 → 训练 → NSGA-II → TOPSIS"""
    print("=" * 60)
    print("  HDP-DS v3.1 动态供应商选择系统")
    print("=" * 60)

    print("\n[1] 加载数据...")
    loader, dataset = _load_dataset(excel_path, window_size)
    loader.print_data_summary()

    print("\n[2] 构建数据集...")
    train_ds, val_ds, test_ds = loader.train_val_test_split(dataset)
    if any(len(split) == 0 for split in (train_ds, val_ds, test_ds)):
        raise ValueError('数据不足以划分非空训练、验证、测试集，请增加时序数据')
    static_train, dynamic_train, kpi_train = train_ds.to_numpy()
    static_val, dynamic_val, kpi_val = val_ds.to_numpy()
    static_test, dynamic_test, kpi_test = test_ds.to_numpy()
    print(f"  样本数: {len(dataset)}, 静态: {static_train.shape}, 动态: {dynamic_train.shape}, KPI: {kpi_train.shape}")

    print("\n[3] 训练 HDP-DS 模型...")
    model = HDPModel(HDPModelConfig(use_static=True, use_dynamic=True, use_attention=True, use_multitask=True))
    model.fit(static_train, dynamic_train, kpi_train, static_val, dynamic_val, kpi_val, n_epochs=n_epochs, batch_size=32)

    print("\n[4] 预测 KPI...")
    kpi_pred = model.predict_kpi(static_test, dynamic_test)
    metrics = EvaluationMetrics()
    metrics.print_metrics(metrics.compute_all(kpi_test, kpi_pred), "预测评估")

    print("\n[5] NSGA-II 优化...")
    optimizer = HDPNSGA2(NSGA2Config(verbose=True))
    supplier_ids = [f"S{i}" for i in range(kpi_pred.shape[0])]
    pareto_solutions, pareto_front = optimizer.optimize(kpi_pred, supplier_ids)
    optimizer.print_summary()

    print("\n[6] TOPSIS 排序...")
    decision_maker = TOPSISDecision()
    rankings = decision_maker.rank_suppliers_from_kpi(kpi_pred, supplier_ids)
    decision_maker.print_ranking_report(rankings)

    print("\n[7] 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, 'hdp_model'))

    print(f"\n{'=' * 60}\n  工作流完成\n{'=' * 60}")
    return {'model': model, 'rankings': rankings, 'metrics': metrics.compute_all(kpi_test, kpi_pred), 'pareto_front': pareto_front}


def run_ablation_experiments(static_data, dynamic_data, kpi_data):
    """运行消融实验"""
    print("=" * 60)
    print("  HDP-DS 消融实验")
    print("=" * 60)

    experiments = AblationConfig.get_default_experiments()
    runner = AblationRunner()
    metrics_eval = EvaluationMetrics()

    def train_func(config):
        model = HDPModel(HDPModelConfig(
            use_static=config.use_static, use_dynamic=config.use_dynamic,
            use_attention=config.use_attention, use_multitask=config.use_multitask
        ))
        model.fit(static_data, dynamic_data, kpi_data, n_epochs=50)
        return model

    def eval_func(model):
        pred = model.predict_kpi(static_data, dynamic_data)
        return {n: r.value for n, r in metrics_eval.compute_all(kpi_data, pred).items()}

    runner.run_all_ablations(experiments, train_func, eval_func)
    return runner.results


def run_comparison_experiments(static_data, dynamic_data, kpi_data):
    """运行模型对比实验"""
    print("=" * 60)
    print("  HDP-DS 模型对比实验")
    print("=" * 60)

    predictions = {}
    print("\n[1/5] HDP-DS...")
    model = HDPModel(HDPModelConfig())
    model.fit(static_data, dynamic_data, kpi_data, n_epochs=80)
    predictions['HDP-DS'] = model.predict_kpi(static_data, dynamic_data)

    print("\n[2/5] XGBoost...")
    try:
        from xgboost import XGBRegressor
        xgb_pred = np.zeros_like(kpi_data)
        for i in range(5):
            xgb = XGBRegressor(n_estimators=100)
            xgb.fit(static_data, kpi_data[:, i])
            xgb_pred[:, i] = xgb.predict(static_data)
        predictions['XGBoost'] = xgb_pred
    except Exception as e:
        print(f"  XGBoost 失败: {e}")

    print("\n[3/5] BP...")
    try:
        from sklearn.neural_network import MLPRegressor
        bp_pred = np.zeros_like(kpi_data)
        for i in range(5):
            bp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)
            bp.fit(static_data, kpi_data[:, i])
            bp_pred[:, i] = bp.predict(static_data)
        predictions['BP'] = bp_pred
    except Exception as e:
        print(f"  BP 失败: {e}")

    print("\n[4/5] LSTM+LR...")
    try:
        from sklearn.linear_model import LinearRegression
        X_flat = dynamic_data[:, -1, :]
        lr = LinearRegression()
        lr.fit(X_flat, kpi_data)
        predictions['LSTM+LR'] = lr.predict(X_flat)
    except Exception as e:
        print(f"  LSTM+LR 失败: {e}")

    predictions['Random'] = np.random.rand(*kpi_data.shape)
    ExperimentComparator().print_comparison_table(kpi_data, predictions)
    return predictions


def run_feedback_demo(static_data, dynamic_data, kpi_data):
    """运行闭环反馈演示"""
    print("=" * 60)
    print("  HDP-DS 闭环反馈演示")
    print("=" * 60)

    model = HDPModel(HDPModelConfig())
    model.fit(static_data, dynamic_data, kpi_data, n_epochs=50)
    optimizer = HDPNSGA2(NSGA2Config())
    decision_maker = TOPSISDecision()

    feedback = FeedbackLoop(
        config=FeedbackConfig(n_cycles=5, simulate_drift=True),
        model=model, optimizer=optimizer, decision_maker=decision_maker
    )
    supplier_ids = [f"S{i}" for i in range(kpi_data.shape[0])]
    supplier_names = [f"供应商_{i}" for i in range(kpi_data.shape[0])]
    results = feedback.run_cycles(static_data, dynamic_data, kpi_data, supplier_ids, supplier_names)
    feedback.print_report()

    print(f"\n  KPI 变化轨迹 (每周期均值):")
    for i, kpi_mean in enumerate(feedback.get_kpi_trajectory()):
        print(f"    周期 {i+1}: {kpi_mean}")
    return results


def _run_demo():
    """使用模拟数据运行演示"""
    print("\n[演示模式] 生成模拟数据...")
    np.random.seed(42)
    n_suppliers, n_timesteps = 20, 24

    static_data = np.random.rand(n_suppliers, 6).astype(np.float32)
    static_data[:, 1] *= 5000
    static_data[:, 5] *= 25

    dynamic_data = np.random.rand(n_suppliers, n_timesteps, 6).astype(np.float32)
    dynamic_data[:, :, 0] *= 100
    dynamic_data[:, :, 1] *= 100
    dynamic_data[:, :, 2] *= 20
    dynamic_data[:, :, 3] *= 5
    dynamic_data[:, :, 4] = (np.random.rand(n_suppliers, n_timesteps) - 0.5) * 20
    dynamic_data[:, :, 5] = np.random.rand(n_suppliers, n_timesteps) * 48

    kpi_data = np.random.rand(n_suppliers, 5).astype(np.float32)
    print(f"  静态: {static_data.shape}, 动态: {dynamic_data.shape}, KPI: {kpi_data.shape}")

    n_train = int(n_suppliers * 0.7)
    s_train, s_test = static_data[:n_train], static_data[n_train:]
    d_train, d_test = dynamic_data[:n_train], dynamic_data[n_train:]
    k_train, k_test = kpi_data[:n_train], kpi_data[n_train:]

    print("\n[演示] 训练 HDP-DS 模型...")
    model = HDPModel(HDPModelConfig(use_static=True, use_dynamic=True, use_attention=True, use_multitask=True))
    model.fit(s_train, d_train, k_train, s_test, d_test, k_test, n_epochs=50, verbose=True)

    kpi_pred = model.predict_kpi(s_test, d_test)
    EvaluationMetrics().print_metrics(EvaluationMetrics().compute_all(k_test, kpi_pred), "预测评估")

    print("\n[演示] NSGA-II 优化...")
    optimizer = HDPNSGA2(NSGA2Config())
    supplier_ids = [f"S{i}" for i in range(kpi_pred.shape[0])]
    optimizer.optimize(kpi_pred, supplier_ids)
    optimizer.print_summary()

    print("\n[演示] TOPSIS 排序...")
    decision_maker = TOPSISDecision()
    decision_maker.print_ranking_report(decision_maker.rank_suppliers_from_kpi(kpi_pred, supplier_ids))

    print(f"\n{'=' * 60}\n  演示完成")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='HDP-DS v3.1', allow_abbrev=False)
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'ablation', 'comparison', 'feedback', 'demo'])
    parser.add_argument('--data', type=str, default=DEFAULT_EXCEL_PATH)
    parser.add_argument('--historical', type=str, default=DEFAULT_HISTORICAL_PATH,
                        help='兼容参数：当前 HDP 流程不读取此文件，KPI 使用 --data 内的工作表')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--window', type=int, default=12)
    args = parser.parse_args()
    if args.epochs <= 0 or args.window <= 0:
        parser.error('--epochs 和 --window 必须是正整数')

    print(f"\nHDP-DS v3.1 — 模式: {args.mode}")

    if args.mode == 'demo':
        _run_demo()
        return

    try:
        if args.mode == 'full':
            return run_full_workflow(args.data, args.historical, args.window, args.epochs)

        _, dataset = _load_dataset(args.data, args.window)
        static_data, dynamic_data, kpi_data = dataset.to_numpy()
        print(f"数据加载成功: {static_data.shape}, {dynamic_data.shape}, {kpi_data.shape}")
        if args.mode == 'ablation':
            return run_ablation_experiments(static_data, dynamic_data, kpi_data)
        if args.mode == 'comparison':
            return run_comparison_experiments(static_data, dynamic_data, kpi_data)
        if args.mode == 'feedback':
            return run_feedback_demo(static_data, dynamic_data, kpi_data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
