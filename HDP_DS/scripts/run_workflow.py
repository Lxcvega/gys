"""HDP-DS v3.1 工作流运行脚本"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



warnings.filterwarnings('ignore')
from collections import Counter

from HDP_DS.data.data_loader import HDPDataLoader
from HDP_DS.models.hdp_model import HDPModel, HDPModelConfig
from HDP_DS.optimization.nsga2 import HDPNSGA2, NSGA2Config
from HDP_DS.optimization.decision import TOPSISDecision
from HDP_DS.evaluation.metrics import EvaluationMetrics


def run(excel_path: str | None = None):
    """执行完整HDP-DS工作流"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'raw')

    print('=' * 60)
    print('  HDP-DS v3.1 完整工作流')
    print('=' * 60)

    print('\n[1] 加载数据...')
    if excel_path is None:
        excel_path = os.path.join(data_dir, 'hdp_ds_mock_data.xlsx')
        if not os.path.exists(excel_path):
            print("  未找到数据文件，将生成虚拟数据...")
            from HDP_DS.data.generation import generate_mock_data
            generate_mock_data(output_path=excel_path)

    loader = HDPDataLoader(window_size=12, forecast_horizon=3)
    loader.load_from_excel(excel_path)
    loader.print_data_summary()

    dataset = loader.build_dataset()
    train_ds, val_ds, test_ds = loader.train_val_test_split(dataset)
    static_train, dynamic_train, kpi_train = train_ds.to_numpy()
    static_val, dynamic_val, kpi_val = val_ds.to_numpy()
    static_test, dynamic_test, kpi_test = test_ds.to_numpy()
    print(f'\n  训练: {static_train.shape[0]}, 验证: {static_val.shape[0]}, 测试: {static_test.shape[0]} 样本')

    print('\n[2] 训练 HDP-DS 模型...')
    model = HDPModel(HDPModelConfig(use_static=True, use_dynamic=True, use_attention=True, use_multitask=True))
    model.static_encoder.fit(static_train, kpi_train)
    model.tft_encoder.fit(dynamic_train, kpi_train, dynamic_val, kpi_val, n_epochs=30)
    model.fit(static_train, dynamic_train, kpi_train, static_val, dynamic_val, kpi_val, n_epochs=20, batch_size=32, verbose=True)

    print('\n[3] 预测评估...')
    kpi_pred = model.predict_kpi(static_test, dynamic_test)
    metrics = EvaluationMetrics()
    results = metrics.compute_all(kpi_test, kpi_pred)
    metrics.print_metrics(results, '预测评估')

    print('\n[4] NSGA-II 优化...')
    optimizer = HDPNSGA2(NSGA2Config())
    supplier_ids = [f'S{i:03d}' for i in range(kpi_pred.shape[0])]
    pareto_solutions, pareto_front = optimizer.optimize(kpi_pred, supplier_ids)
    optimizer.print_summary()

    print('\n[5] TOPSIS 排序...')
    decision_maker = TOPSISDecision()
    rankings = decision_maker.rank_suppliers_from_kpi(kpi_pred, supplier_ids)
    decision_maker.print_ranking_report(rankings)

    print('\n[6] 静态特征重要性:')
    importance = model.static_encoder.get_feature_importance()
    for name, val in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f'  {name:22s} {val:>8.4f}')

    print(f'\n{"=" * 60}')
    print(f'  工作流完成!  MAE={results["MAE"].value:.4f}  RMSE={results["RMSE"].value:.4f}  R2={results["R2"].value:.4f}')
    if rankings:
        lc = Counter(r.recommendation_level for r in rankings)
        print(f'  TOPSIS推荐: {dict(lc)}  最佳: {rankings[0].supplier_name} (贴近度={rankings[0].closeness:.4f})')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HDP-DS v3.1 工作流')
    parser.add_argument('--excel', type=str, default=None)
    args = parser.parse_args()
    run(args.excel)
