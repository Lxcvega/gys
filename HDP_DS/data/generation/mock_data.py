"""
HDP-DS v3.1 虚拟数据生成器
基于行业公开数据规律生成真实的供应商模拟数据

用法:
  from HDP_DS.data.generation import generate_mock_data
  df_static, df_dynamic, df_kpi = generate_mock_data()
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional

# 季节性模式 (所有供应商共享)
MONTHLY_PATTERN = {
    1: -0.05, 2: -0.03, 3: 0.00, 4: 0.02, 5: 0.03,
    6: 0.05, 7: 0.08, 8: 0.06, 9: 0.03, 10: 0.00,
    11: -0.02, 12: -0.05
}

# 供应商基本信息 (参考中国环保行业公开数据)
DEFAULT_SUPPLIERS = [
    ('S001', '碧水源科技', 120000, 1, 1, 3, 2001),
    ('S002', '首创环保集团', 880000, 1, 1, 3, 1999),
    ('S003', '中持股份', 50000, 1, 1, 2, 2009),
    ('S004', '博天环境', 40000, 1, 0, 2, 1995),
    ('S005', '国祯环保', 70000, 1, 1, 2, 1997),
    ('S006', '中原环保', 300000, 1, 1, 3, 2007),
    ('S007', '武汉控股', 150000, 0, 0, 2, 1998),
    ('S008', '洪城环境', 100000, 1, 1, 2, 2001),
    ('S009', '重庆水务', 480000, 1, 1, 3, 2001),
    ('S010', '兴蓉环境', 300000, 1, 1, 2, 1996),
    ('S011', '江南水务', 100000, 0, 0, 1, 2003),
    ('S012', '绿城水务', 90000, 1, 0, 2, 2006),
    ('S013', '中山公用', 250000, 1, 1, 2, 2001),
    ('S014', '鹏鹞环保', 80000, 1, 0, 1, 1997),
    ('S015', '中环环保', 40000, 0, 0, 1, 2011),
    ('S016', '联泰环保', 60000, 1, 1, 2, 2006),
    ('S017', '上海环境', 450000, 1, 1, 3, 2004),
    ('S018', '绿色动力', 350000, 1, 1, 2, 2000),
    ('S019', '伟明环保', 170000, 1, 1, 2, 2005),
    ('S020', '清新环境', 140000, 0, 0, 1, 2001),
]


def generate_mock_data(
    suppliers: Optional[list] = None,
    start_year: int = 2023,
    n_months: int = 24,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    生成 HDP-DS 虚拟供应商数据

    Args:
        suppliers: 供应商列表，每项 (id, name, capital, iso, env, credit, founding_year)
        start_year: 起始年份
        n_months: 月数
        random_seed: 随机种子
        output_path: Excel 输出路径 (None=不保存)

    Returns:
        (df_static, df_dynamic, df_kpi): 静态/动态/KPI 三张表
    """
    np.random.seed(random_seed)

    if suppliers is None:
        suppliers = DEFAULT_SUPPLIERS

    # ── 静态数据 ──
    static_data = []
    for sid, name, cap, iso, env, credit, year in suppliers:
        age = start_year + n_months // 12 - year
        scale = np.log10(cap) / np.log10(1_000_000)
        static_data.append({
            'supplier_id': sid,
            'supplier_name': name,
            'enterprise_scale': round(scale, 4),
            'registered_capital': float(cap),
            'iso_certified': iso,
            'env_certified': env,
            'credit_rating': credit,
            'enterprise_age': age,
        })
    df_static = pd.DataFrame(static_data)

    # ── 动态时序 ──
    periods = [f"{y}-{m:02d}" for y in range(start_year, start_year + n_months // 12 + 1)
               for m in range(1, 13)][:n_months]
    dynamic_records = []

    for sid, name, cap, iso, env, credit, year in suppliers:
        base_level = 0.55 + credit * 0.10
        if iso: base_level += 0.05
        if env: base_level += 0.03
        volatility = 0.15 - credit * 0.03

        for t, period in enumerate(periods):
            month = int(period.split('-')[1])
            season = MONTHLY_PATTERN[month]
            drift = 0.002 * t * (1 if credit >= 2 else -0.5)
            shock = np.random.randn() * volatility

            delivery_rate = np.clip(base_level * 100 + season * 20 + drift * 10 + shock * 5, 70, 100)
            quality_rate = np.clip(base_level * 100 + season * 5 + drift * 5 + shock * 3, 75, 100)
            complaint_count = round(np.clip(max(0, (1 - base_level) * 8 + abs(shock) * 2 - drift * 2), 0, 15), 1)
            env_rate = max(0, (1 - base_level) * 0.5 - env * 0.2)
            env_incident = np.random.poisson(env_rate)
            cost_change = np.clip(np.random.randn() * 3 + season * 2 + drift * 0.5, -15, 15)
            service_time = np.clip(48 - base_level * 30 + abs(shock) * 4 + season * 5, 2, 72)

            dynamic_records.append({
                'supplier_id': sid, 'period': period,
                'delivery_rate': round(delivery_rate, 1),
                'quality_rate': round(quality_rate, 1),
                'complaint_count': complaint_count,
                'env_incident_count': int(env_incident),
                'cost_change_rate': round(cost_change, 2),
                'service_response_time': round(service_time, 1),
            })
    df_dynamic = pd.DataFrame(dynamic_records)

    # ── KPI 标签 ──
    kpi_records = []
    for sid, name, cap, iso, env, credit, year in suppliers:
        base_level = 0.55 + credit * 0.10
        if iso: base_level += 0.05
        if env: base_level += 0.03
        volatility = 0.15 - credit * 0.03

        for t, period in enumerate(periods):
            month = int(period.split('-')[1])
            season = MONTHLY_PATTERN[month]
            drift = 0.002 * t * (1 if credit >= 2 else -0.5)
            shock = np.random.randn() * volatility

            p1 = np.clip(base_level + season * 0.15 + drift + shock * 0.05, 0.3, 1.0)
            p2 = np.clip(base_level + drift * 0.5 + np.random.randn() * 0.05, 0.3, 1.0)
            env_risk_rate = max(0, (1 - base_level) * 0.4 - env * 0.15)
            p3 = np.clip(env_risk_rate + abs(shock) * 0.1, 0.0, 0.8)
            p4 = np.clip(max(0.1, 1.0 - abs(np.random.randn() * 0.05 + season * 0.03)), 0.1, 0.95)
            p5 = np.clip(base_level + drift * 0.3 - season * 0.1 + np.random.randn() * 0.05, 0.2, 1.0)

            kpi_records.append({
                'supplier_id': sid, 'period': period,
                'delivery_reliability': round(p1, 4),
                'quality_stability': round(p2, 4),
                'environmental_risk': round(p3, 4),
                'cost_stability': round(p4, 4),
                'service_capability': round(p5, 4),
            })
    df_kpi = pd.DataFrame(kpi_records)

    # ── 保存 ──
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_static.to_excel(writer, sheet_name='静态数据', index=False)
            df_dynamic.to_excel(writer, sheet_name='动态数据', index=False)
            df_kpi.to_excel(writer, sheet_name='KPI标签', index=False)
        print(f"[生成] 数据已保存到: {output_path}")
        print(f"       静态: {len(df_static)} 供应商, 动态: {len(df_dynamic)} 行, KPI: {len(df_kpi)} 行")

    return df_static, df_dynamic, df_kpi


if __name__ == '__main__':
    # 独立运行：生成并保存到默认路径
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw')
    output_path = os.path.abspath(os.path.join(base_dir, 'hdp_ds_mock_data.xlsx'))
    generate_mock_data(output_path=output_path)
    print("生成完成!")
