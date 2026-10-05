"""
HDP-DS v3.1 CLI 入口 — 支持 python -m HDP_DS

用法:
  python -m HDP_DS                    # 完整工作流 (默认)
  python -m HDP_DS --mode ablation    # 消融实验
  python -m HDP_DS --mode comparison  # 模型对比
  python -m HDP_DS --mode feedback    # 闭环反馈
  python -m HDP_DS --mode demo        # 演示模式
"""

from HDP_DS.main import main

if __name__ == '__main__':
    main()
