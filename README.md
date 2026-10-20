# 海上风电功率预测：ERA5、GFS 与 LSTM

使用气象数据、物理特征和 LSTM 估计单台 5 MW 海上参考风机的逐小时功率，输出未来第 1—24 小时的 24 个功率点。仓库包含已训练模型、训练代码、小型复现数据、评价与气象输入影响分析，以及气象下载和 OpenFAST 标签生成接口。

更新日期：2026-10-20。

## 当前进展

已完成 ERA5 条件功率模型、GFS 预报功率模型，以及后续的全年天气订正和功率订正模型。以下主模型比较统一使用 **2024 年四季各连续 7 天，共 28 次每日起报、672 个不同目标小时**。

| 气象输入与方法 | RMSE / kW | MAE / kW | 相对同气象经验公式的 RMSE 改善 |
|---|---:|---:|---:|
| ERA5＋经验公式 | 181.07 | 106.47 | — |
| ERA5＋条件 LSTM | 109.66 | 66.52 | 39.43% |
| GFS＋经验公式 | 1164.88 | 703.29 | — |
| GFS＋LSTM | 1115.59 | 731.04 | 4.23% |

原报告 ERA5 改善 **40.07%** 使用 580 个重叠窗口、13920 个计分点；上表改成与 GFS 相同的每日窗口，因此为 **39.43%**。GFS 主模型的 RMSE 改善，但 MAE 增大。

后续 v5 使用 0.25°、原生逐小时 GFS 和 2023 全年训练资料。在同一批已使用的 2024 参考功率上：

| 全年订正模型 | RMSE / kW | MAE / kW |
|---|---:|---:|
| 天气订正 LSTM | 1087.98 | 666.34 |
| 功率订正 LSTM | 1052.44 | 683.20 |

**提前预测的 RMSE 尚未低于目标时刻 ERA5＋经验公式的 181.07 kW 对照。** ERA5 条件模型使用事后目标时刻气象；GFS 模型使用起报时已发布的预报。参考功率来自 ERA5 驱动的 OpenFAST 仿真，每个工况运行 720 秒、去掉 120 秒预热，取后 600 秒发电机功率均值。研究对象为 NREL 5 MW OC3 单桩参考风机，固定湍流强度 8%；参考标签不是实测风场 SCADA。后续订正与归因使用过这批 2024 数据，其成绩属于开发评价。

## 安装与快速运行

Python 3.11—3.13，CPU 即可运行模型。以下命令均在仓库根目录执行。

```bash
git clone https://github.com/Lonelylone0328/offshore-wind-lstm.git
cd offshore-wind-lstm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -c requirements-repro.txt
```

先运行随仓库提供的 24 小时样例：

```bash
windpower predict --model gfs --input examples/gfs_24h.csv --output runs/gfs.csv
windpower predict --model era5 --input examples/era5_24h.csv --output runs/era5.csv
windpower predict --model weather --input examples/native_24h.csv --output runs/weather.csv
windpower predict --model power --input examples/native_24h.csv --output runs/power.csv
```

结果包含 `issue_time`、`target_time`、`lead_hours` 和 `power_kw`。时间统一为 UTC，功率单位为 kW。样例是历史资料重放。默认读取 `models/<模型名>/`；使用新训练模型时传入 `--model-dir runs/my_model`。

`python -m windpower` 与 `windpower` 等价。输出放在 `runs/`，同名输出会被更新。

## 代码结构

```text
src/windpower/
  physics.py              物理特征与经验功率曲线
  datasets.py             气象特征与功率标签配对、训练窗口构造
  models.py               ERA5/GFS 主模型及集成推理
  training.py             主模型训练
  correction.py           145维 GFS 特征及全年订正推理
  correction_training.py  全年天气/功率订正训练
  analysis.py             六组交叉输入评价和误差分解
  acquisition.py          ERA5/GFS 下载与预处理
  simulation.py           OpenFAST/TurbSim 工况和功率统计
  cli.py                  统一命令行入口
models/                   14份选定权重、标准化参数和成员清单
data/                     2023训练、2024评价及数值对照数据
examples/                 可直接运行的输入样例
results/                  固定参考成绩与逐点预测
scripts/                  下载、OpenFAST安装和仿真入口
tests/                    模型复现、时间配对和数值分解测试
docs/                     模型结构、输入格式、数据与仿真说明
```

## 四种模型

| 命令中的模型名 | 输入 | 方法与成员数 |
|---|---|---|
| `era5` | 24×4 个目标时刻 ERA5 物理特征 | 单层 8 隐单元残差 LSTM，3 成员均值 |
| `gfs` | 24×4 个已发布 GFS 物理特征 | 单层 8 隐单元残差 LSTM，5 成员均值 |
| `weather` | 48小时×18点×7个原始 GFS 字段 | 145维派生输入，双向 16 隐单元天气订正，3 成员均值后接固定 ERA5 功率模型 |
| `power` | 与 `weather` 相同 | 双向 16 隐单元功率残差订正，3 成员均值 |

四项物理特征为 90 m 轮毂风速、空气密度、风切变指数和经验功率。主模型的每个成员有 457 个参数。

经验公式使用固定分段三次曲线：切入 3 m/s、额定风速 11.4 m/s、切出 25 m/s、额定功率 5000 kW。在运行区间内，

```text
P(v) = 5000 × clip((v³ − 3³) / (11.4³ − 3³), 0, 1)
```

风速低于切入或达到切出时功率为零。ERA5/GFS 主模型在经验功率上学习残差；两套权重采用各自已保存的标准化参数。全年功率订正模型按原方法将单成员输出限制在 0—5000 kW 后平均。详细结构和字段见 [模型说明](docs/models.md)。

## 复现评价与气象影响分析

```bash
windpower evaluate --output runs/evaluation
windpower evaluate-corrections --output runs/corrections
```

第一个命令固定两套主模型、交叉输入 ERA5 和 GFS，生成六组预测：

| 固定方法 | ERA5 输入 RMSE / kW | GFS 输入 RMSE / kW |
|---|---:|---:|
| 经验公式 | 181.07 | 1164.88 |
| ERA5 训练的 LSTM | 109.66 | 1140.96 |
| GFS 训练的 LSTM | 303.94 | 1115.59 |

输出包括 `predictions.csv`、`summary.json`、`grouped_results.json`，以及 `metrics_by_scope.csv`、`attribution_by_scope.csv`、`mse_by_scope.csv`、`weather_difference_by_scope.csv`。覆盖总体、4 季、28 日、24 个预测时效和留一季，共 61 个评价范围。

也可以分析已经保存的六组预测：

```bash
windpower attribute --input results/reference_predictions.csv --output runs/attribution
```

分解结果：

- 统一口径后的改善率差为 **35.20 个百分点**。对称两因素分配得到气象来源项 **−17.36**、冻结模型项 **52.56** 个百分点。
- 从绝对 RMSE 节省与比较分母的另一角度分解，分别为 **7.05** 和 **28.15** 个百分点。分母项约占 79.96%，这是指标分母效应。
- 固定方法的 MSE 分解为 `MSE_G = MSE_E + mean(d²) + 2×mean(e×d)`，其中 `e=f(ERA5)−y`，`d=f(GFS)−f(ERA5)`。

这两种改善率分解是同一差距的不同描述，不能重复相加。气象来源项含输入分辨率、插值及基准变化；模型项含训练方案与成员差异。来源项为负体现相对改善率的交互作用，两套固定 LSTM 在 ERA5 输入下的绝对误差仍都更低。由于缺少独立实测气象、标签又由 ERA5 驱动生成，这些量不能解释为相对真实天气的误差占比。

原始窗口与每日窗口的口径衔接记录保存在 [参考汇总](results/reference_summary.json) 的 `historical_comparison` 中。

## 重新训练

使用随仓库提供的数据，训练命令为：

```bash
windpower train --model era5 --data data/era5_train.npz --output runs/train_era5
windpower train --model gfs --data data/gfs_train.npz --output runs/train_gfs
windpower train --model weather --data data/native_train.npz --targets data/native_targets.npz --output runs/train_weather
windpower train --model power --data data/native_train.npz --targets data/native_targets.npz --output runs/train_power
```

- ERA5：2023 冬春训练、夏季验证、秋季留出；按验证误差早停，最多 600 轮，3 个固定种子。
- GFS 主模型：用已选定配置对 2023 四季共 12 个每日窗口训练，5 个种子各 23 轮。
- 全年订正：2023 年 293 日训练、60 日验证、12 日隔离，3 个种子。ERA5 仅提供训练监督；推理只读取已发布 GFS。天气和功率订正最多各 500 轮。

标准化仅拟合训练数据。训练入口保留最终选定配置，删除了旧版候选搜索。快速检查训练流程可追加 `--epochs 1`；它生成的模型不对应上面的正式成绩。训练输出目录应与 `models/` 分开。

## 使用自己的气象与功率资料

原始气象字段为 `time,site_id,u10,v10,u100,v100,t2m,d2m,sp`：风速分量用 m/s，温度与露点用 K，压力用 Pa。`site_id=sea` 为功率计算点。GFS 另外带起报、有效时间、循环和可用时间字段。

先计算物理特征，再与 `time,target_power_kw,segment_id` 功率标签组装训练窗口：

```bash
windpower prepare --input data/era5_2023.csv --output runs/era5_features.csv
windpower build-dataset --model era5 --features runs/era5_features.csv --labels data/openfast_2023.csv --output runs/era5_train.npz
windpower prepare --input data/gfs_raw_2023.csv --output runs/gfs_features.csv
windpower build-dataset --model gfs --features runs/gfs_features.csv --labels data/openfast_2023.csv --output runs/gfs_train.npz
```

ERA5 按连续时段构建 24 小时滚动窗口；GFS 按每次起报构建 24 小时窗口。`segment_id` 使用 `winter,spring,summer,autumn`，ERA5 训练/验证片段可用 `--train-segments`、`--validation-segments` 指定。

全年订正的 ERA5 监督可重新对齐生成：

```bash
windpower build-targets --data data/native_train.npz --era5 data/era5_2023.csv --output runs/native_targets.npz
```

`data/era5_train.npz` 与 `data/benchmark.npz` 保留原实验的数值精度，是复现已列成绩的输入。重新组装 ERA5 训练集使用 CSV 标签的完整精度，与历史 float32 标签最多相差 0.000242 kW；从原始气象 CSV 重建特征也可能有微小浮点差异。

## 下载气象与生成 OpenFAST 标签

已有数据足够运行预测、训练和评价。需要增加资料时安装下载依赖：

```bash
python -m pip install -e '.[download]'
python scripts/download_weather.py --help
python scripts/setup_openfast.py --dry-run
python scripts/simulate.py --help
```

下载器提供 ERA5 点位资料、0.5°/3小时 GFS 和 0.25°/逐小时 GFS。OpenFAST 安装脚本固定上游提交，使用配套 NREL OC3 模型与 DISCON 控制器。第三方源码、编译产物、原始风场与完整仿真输出在本地生成，不随仓库上传。

具体下载日期、点位、OpenFAST 安装及单工况/批量仿真命令见 [数据与仿真说明](docs/data-and-simulation.md)。

## 验证

```bash
python -m pytest -q
```

测试覆盖物理计算、主模型与全年订正模型的逐点复现、训练权重重载、24小时时间配对、六组评价和误差分解。发布整理时，主模型四格推理及两套主模型完整重训均与原结果逐点一致；全年订正推理差异小于 `1e-9 kW`。仿真接口通过固定模板工况准备和已有 OpenFAST 输出统计复核，未重新运行整批仿真。

验证环境：Python 3.12、PyTorch 2.14.0、NumPy 2.5.2、pandas 3.0.5，CPU。`requirements-repro.txt` 固定这三项计算依赖的验证版本。

## 数据来源

- ERA5：[ECMWF 单层再分析数据](https://doi.org/10.24381/cds.adbb2d47)，通过 Earthmover 公共镜像提取海陆点。
- GFS：[NOAA GFS 公共归档](https://registry.opendata.aws/noaa-gfs-bdp-pds/)，按起报时可用的循环和对象提取。
- 仿真：[OpenFAST](https://github.com/OpenFAST/openfast) 与 [r-test 参考模型](https://github.com/OpenFAST/r-test)。固定提交和第三方许可见 [数据与仿真说明](docs/data-and-simulation.md)。

成员与数据的来源记录见 [SOURCES.json](models/SOURCES.json)。
