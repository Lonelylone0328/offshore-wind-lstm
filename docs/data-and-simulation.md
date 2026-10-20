# 数据下载与仿真标签

仓库自带的数据可直接运行训练、预测和评价。以下接口用于新增时段的数据；下载缓存、OpenFAST 源码、编译文件和仿真风场由本机保存。

## ERA5

ERA5 下载使用 Earthmover 的公共 Icechunk 镜像，提取两个固定格点：海上 33°N、122°E，陆地 33°N、120°E。

```bash
pip install -e '.[download]'
python scripts/download_weather.py era5 \
  --start 2024-01-15T00:00:00Z --end 2024-01-15T23:00:00Z \
  --output data/new-era5.csv
```

输出 CSV 的基本列为：

| 列 | 单位或含义 |
|---|---|
| `time` | UTC 有效时间 |
| `site_id,latitude,longitude` | 格点标识与坐标 |
| `u10,v10,u100,v100` | 10 m、100 m 风速分量，m/s |
| `t2m,d2m` | 2 m 温度、露点，K |
| `sp` | 地面气压，Pa |
| `lsm` | 陆地比例 |

同名 `.source.json` 保存镜像快照、数据集 DOI 和单位。用 `--snapshot` 指定已有记录的快照；用 `--sites sites.csv` 更换格点，文件列为 `site_id,latitude,longitude`，坐标需在原始网格上。

## GFS

起报清单是包含 `issue_time` 列的 CSV。项目采用目标日前一天 23 UTC 起报：

```csv
issue_time
2024-01-14T23:00:00Z
```

保存为 `data/issues.csv` 后，下载报告模型使用的 0.5° 数据：

```bash
python scripts/download_weather.py gfs \
  --issue-times data/issues.csv --resolution 0.5 \
  --output data/new-gfs.csv
```

该模式从起报前至少 6 小时的完整 GFS 周期中选取未来 24 小时。原生时间间隔为 3 小时，先对风速分量、温度、露点、气压做线性插值，再计算物理特征。插值两端的源对象、权重和可用时间均保存在输出中。

全年订正模型使用 0.25°、原生逐小时、海陆各 3×3 邻域，输入为过去 24 小时加未来 24 小时：

```bash
python scripts/download_weather.py gfs \
  --issue-times data/issues.csv --resolution 0.25 --neighbors --history-hours 24 \
  --output data/new-native.csv --npz-output data/new-native.npz
```

这一模式优先使用当日 18 UTC 周期。历史时刻早于所选周期时，使用前一天相同 UTC 周期的预报。全部源对象必须在起报前 15 分钟已可用；否则整次起报按 6 小时向前选择周期，最多回退 48 小时。`--minimum-cycle-age` 和 `--buffer-minutes` 可调整这两个参数。

GFS CSV 除七个气象变量和坐标，还包含 `issue_time,time,target_time,valid_time,lead_hours,cycle_time,forecast_hour,available_time`。`time`、`target_time`、`valid_time` 为同一有效时间，分别兼容物理特征、功率预测和全年订正入口。历史部分的 `lead_hours` 为 −23…0，预测部分为 1…24。

NPZ 保留原始订正接口：`raw_values` 为 `[起报数,48,18,7]`，另有 `raw_columns,points_lat_lon,issue_times,valid_times,cycle_times,available_times,fallback_hours`。邻域顺序为海上、陆地，随后纬度偏移 −0.25、0、0.25，再按经度偏移相同顺序排列。

下载按 GRIB 索引请求七个字段的字节范围，用 ecCodes 解码原始格点；`--cache` 指定本地缓存，`--workers` 控制并发。`available_time` 使用 S3 数据对象与索引的最大 `Last-Modified`，它是档案上传时间代理。

## 物理特征

```bash
python - <<'PY'
import pandas as pd
from windpower.physics import build_features

raw = pd.read_csv("data/new-era5.csv")
build_features(raw).to_csv("data/new-features.csv", index=False)
PY
```

输出包含 `time,hub_wind_speed_mps,air_density_kgm3,shear_exponent,theoretical_power_kw`。GFS 有多个起报时，按 `issue_time` 分组后分别调用，避免把不同起报的相同有效时间合并。

## OpenFAST 安装

使用以下固定版本：

| 组件 | 提交 |
|---|---|
| [OpenFAST](https://github.com/OpenFAST/openfast) | `2895884d2be01862173c88d70f86b358d2f1a50a` |
| [参考模型 r-test](https://github.com/OpenFAST/r-test) | `dd5feaaaa500ba7283140107806300d551cff0a7` |

先在系统中安装 Git、CMake、GNU Fortran 和 BLAS/LAPACK，然后执行：

```bash
python scripts/setup_openfast.py --dry-run
python scripts/setup_openfast.py --jobs 4
```

Linux 可使用发行版的 `gfortran`、`cmake`、`libopenblas-dev`。macOS 可用 Homebrew 安装 `gcc cmake openblas`，并按安装路径传入：

```bash
python scripts/setup_openfast.py --compiler gfortran \
  --cmake-arg=-DBLA_VENDOR=OpenBLAS \
  --cmake-arg="-DCMAKE_PREFIX_PATH=$(brew --prefix openblas)"
```

`--prefix` 可将源码和构建目录放在仓库之外。脚本使用 GNU Fortran 的共享库参数，适用于 Linux/macOS；其他编译工具链可自行构建后通过仿真入口指定文件路径。

目录布局：

```text
vendor/
  openfast/          OpenFAST 源码及 TurbSim 模板
  r-test/            配套参考机组、翼型和控制器源码
  build/             OpenFAST、TurbSim 编译结果
  DISCON.dll         本机编译的控制器共享库
```

控制器明确取自 `r-test/glue-codes/openfast/5MW_Baseline/ServoData/DISCON/DISCON.F90`。它的桨距限速基于上一步命令，与本机组模板配套。`DISCON.dll` 是固定文件名，文件内容由当前系统编译。

## 生成标签

先准备一个工况的输入文件：

```bash
python scripts/simulate.py --features data/new-features.csv \
  --output-dir runs/simulation --limit 1 --prepare-only
```

确认安装路径后，运行一例或批量运行：

```bash
python scripts/simulate.py --features data/new-features.csv \
  --output-dir runs/simulation --limit 1

python scripts/simulate.py --features data/new-features.csv \
  --output-dir runs/simulation --workers 4
```

输入采用上述物理特征 CSV，每小时一行，可额外提供 `case_id,seed,turbulence_intensity`。默认种子为 `20260906 + 行序号`，TI 为 0.08；需要复用已有工况时显式提供原种子。

运行设置固定为：

- NREL 5 MW OC3 单桩，轮毂 90 m，转子直径 126 m。
- TurbSim IEC Kaimal，15×15 点、170 m×170 m，风场步长 0.1 s；实际高度为 5–175 m。
- OpenFAST 720 s，积分步长 0.01 s，输出步长 0.1 s，剔除前 120 s。
- 无浪，平均来风与机舱对齐；平均风速不在 `[3,25)` m/s 时采用锁定转子、90° 羽化的停车工况。
- 标签为剩余 600 s 内 `GenPwr` 的平均值，单位 kW。

代码检查实际 BTS 网格与平移风场的时间覆盖，保留完整积分、正常结束和控制器运行范围检查。每个小时独立初始化，标签表示该小时气象条件下的 10 分钟代表性电功率。

输出 `labels.csv` 含 `time,case_id,target_power_kw,power_std_kw,half_mean_difference_kw`。每例目录保留模型输入、风场、输出、日志及 `result.json`，同一配置完成后再次运行会复用结果。最后一列是有效段前后两半的功率均值差，可用于观察稳定程度。

已有安装可通过 `--installation` 使用相同布局，或分别指定 `--openfast --turbsim --controller --templates --turbsim-template`。`--templates` 指向 r-test 的 `glue-codes/openfast` 目录。

## 上游来源

ERA5 原始数据集为 [ERA5 hourly data on single levels](https://doi.org/10.24381/cds.adbb2d47)，本下载器使用 [Earthmover 公共镜像](https://registry.opendata.aws/earthmover-era5/)。GFS 来自 [NOAA Open Data](https://registry.opendata.aws/noaa-gfs-bdp-pds/)。

OpenFAST 与 r-test 以 Apache-2.0 发布。构建脚本下载完整上游仓库并保留其 `LICENSE` 和控制器源码中的版权声明，本仓库不内置第三方源码或二进制。机组参数参考 [NREL 5 MW 参考风机报告](https://docs.nrel.gov/docs/fy09osti/38060.pdf)。
