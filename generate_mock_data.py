import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HDP_DS.data.generation import generate_mock_data

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    output_path = os.path.join(output_dir, 'hdp_ds_mock_data.xlsx')
    os.makedirs(output_dir, exist_ok=True)
    generate_mock_data(output_path=output_path)
    print("生成完成!")