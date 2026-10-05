# 污水供应商选择系统

研究用途的供应商评分与选择程序，保留两条流程：默认的交互式 legacy，以及通过 `--hdp` 启动的 HDP-DS。真实业务数据、模型权重、运行报告和密钥不随仓库发布。

## 安装

建议使用 Python 3.11，在项目目录创建虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS 激活命令为 `source .venv/bin/activate`。核心依赖包含 Excel 读写所需的 openpyxl 和 Word 导出所需的 python-docx。

HDP-DS 还需要 PyTorch、XGBoost 和 pymoo：

```powershell
python -m pip install torch xgboost pymoo
```

`requirements_hdp_ds.txt` 是更广的研究依赖清单，另含 pytorch-lightning、pytorch-forecasting、SHAP 等扩展。当前 HDP 主模型直接使用 PyTorch；pymoo 缺失时现有优化模块会提示并使用简化实现，应记录实际使用的实现。

## 入口与帮助

```powershell
python main.py --help
python main.py --hdp --help
python main.py
python main.py --legacy
```

默认入口和 `--legacy` 都启动交互式传统流程。`--hdp` 与 `--legacy` 不能同时使用；legacy 不接受 HDP 的参数。帮助命令只显示说明，不训练模型。

## Legacy 数据与流程

默认数据路径始终相对于项目根目录解析：

- `data/raw/供应商数据.xlsx`：第一张工作表包含当前供应商指标。
- `data/raw/历史训练数据_模板.xlsx`：历史特征和标签；`历史可靠性` 为 0/1 分类标签，`实际评分` 为 0–100 连续回归标签，至少提供一种。实际评分不添加随机噪声。

供应商表必需列：`供应商编号`、`药剂单价(元/吨)`、`运输成本(元/吨)`、`付款周期(天)`、`药剂有效成分含量(%)`、`供货及时性(%)`、`库存能力(吨)`、`交付能力(1-5级)`、`应急供应能力(1-5级)`、`安全管理水平(1-5级)`。`供应商名称`、联系方式、认证和 `处理污泥效果达标率(%)` 为可选列。

当前供应商读取器先检查原始行：空编号、重复编号（去除首尾空格后）、非数值、无穷值、负成本/天数/库存、超出 0–100 的百分数、非 1–5 整数等级均提示行号并排除。所有重复编号行均排除，避免字典覆盖。每次读取开始即清空缓存，失败不会返回上次数据。真正空白的数值单元格保留已有中位数填充；整列缺失时使用 `data_processor.py` 中的行业参考值并提示，缺失等级填充后取整。有效行仍经过原有异常值截尾，结果需要业务复核；历史训练加载器是独立路径，尚未全面复用这些校验。

可用已有函数生成 **legacy 格式的虚构示例**，不含历史标签：

```powershell
python -c "from pathlib import Path; from excel_data_reader import create_mock_excel_template; Path('data/raw').mkdir(parents=True, exist_ok=True); create_mock_excel_template('data/raw/供应商数据.xlsx')"
```

该命令会覆盖同名文件；请先保留真实业务数据。历史标签需要自行准备，不能把示例数据当作真实训练证据。流程依次执行评分、模糊决策、NN 初筛、NSGA-II、动态博弈和最终选择；历史样本不足时会跳过相关训练。legacy 的 TXT、Excel、Word 报告默认写入运行时的当前目录。

## HDP-DS 数据与运行

HDP 格式与 legacy 格式不同。使用仓库现有生成器创建三张表的 **模拟 Excel**：

```powershell
python generate_mock_data.py
python main.py --hdp --mode full --data data/raw/hdp_ds_mock_data.xlsx --epochs 10 --window 12
```

生成器写入 `data/raw/hdp_ds_mock_data.xlsx`，重复执行会覆盖同名文件。示例内的企业名称及数值是生成器演示素材，不代表经核验的企业业绩或采购推荐。

HDP 工作簿使用 `静态数据`、`动态数据`、`KPI标签` 三张表。静态列：`supplier_id`、`enterprise_scale`、`registered_capital`、`iso_certified`、`env_certified`、`credit_rating`、`enterprise_age`；动态列：`supplier_id`、`period`、`delivery_rate`、`quality_rate`、`complaint_count`、`env_incident_count`、`cost_change_rate`、`service_response_time`；KPI 列：`supplier_id` 以及 `delivery_reliability`、`quality_stability`、`environmental_risk`、`cost_stability`、`service_capability`。具体单位、目标定义和时间对应关系见 `HDP_DS/data/data_loader.py` 与生成器。

```powershell
# 显式随机模拟演示（训练 50 轮）
python main.py --hdp --mode demo

# 以下模式使用 --data 指定的数据
python main.py --hdp --mode ablation --data data/raw/hdp_ds_mock_data.xlsx
python main.py --hdp --mode comparison --data data/raw/hdp_ds_mock_data.xlsx
python main.py --hdp --mode feedback --data data/raw/hdp_ds_mock_data.xlsx
```

`--epochs` 当前只控制 full 模式，其他实验/演示仍使用源码中的固定训练轮数。`--window` 控制 Excel 数据的滚动窗口；显式 demo 使用独立的模拟数组。`--historical` 为兼容参数，当前 HDP 主流程不读取它。

full 默认读取 `data/raw/供应商数据.xlsx`，因此使用 HDP 数据时建议始终传 `--data`。文件缺失、读取失败、无可用滚动样本或特征形状/非有限数值错误会终止并返回错误，不会自动转为随机演示。完整流程只加载一次数据，模型写入项目下 `data/models/hdp_model`。

## 研究边界

本次改动聚焦入口、输入校验、路径、标签保真和发布整理，未改动核心研究算法，也没有新测得的 R² 或优化性能提升。

- HDP 的滚动样本可能来自同一家供应商且窗口重叠；现有时间划分不足以证明无泄漏，正式评估需检查预测跨度、时间隔离和标签对齐。
- 缺少静态表/字段时现有加载器使用默认静态特征；缺少完整 KPI 标签时可能从未来动态数据生成代理目标。这些结果不能等同于独立真实标签的预测质量，应提供完整工作表并复核目标含义，特别是环境风险方向。
- full 的排名对象目前是测试窗口样本，编号为 `S0`、`S1` 等，不是按原始供应商汇总后的唯一排名；不能直接作为企业采购名单。
- ablation 和 comparison 当前在同一批数据上训练与评估。comparison 中名为 `LSTM+LR` 的分支实际上仅对最后时间步拟合线性回归，不能作为真实 LSTM 基线证据。
- 固定 Python/NumPy 种子不保证 PyTorch、设备和依赖版本间结果完全一致；需记录完整输入、依赖、设备和划分，并以独立留出集验证结论。

## 验证与发布

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions 运行同一回归命令并安装核心依赖；回归检查不等同于完整模型训练或科学结论验证。

`.gitignore` 排除 `config/secret_key`、`.env`、`data/raw/`、模型、报告、备份和编辑器目录。发布前检查待提交文件，忽略规则不会自动移除已被 Git 跟踪的敏感文件。请勿上传真实供应商或联系人数据；需要共享示例时优先提供生成命令。
