# LIBERO-Plus 评测

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)（[论文](https://arxiv.org/abs/2510.13626)）是 LIBERO 的鲁棒性扩展基准。它保留原有的 4 个套件（Spatial、Object、Goal、Long/10），并将每个套件扩展为 7 类扰动，共 **10,030 个任务**：

| 官方列 | 扰动类型 | 任务数（4 个套件） |
|---|---|---|
| Camera | 相机视角 | 1,599 |
| Robot | 机器人初始状态 | 1,550 |
| Language | 指令语义改写 | 1,537 |
| Light | 光照条件 | 1,142 |
| Background | 背景纹理 | 1,076 |
| Noise | 传感器噪声 | 1,601 |
| Layout | 物体布局 | 1,525 |
| **合计** | | **10,030** |

官方协议对**每个任务只评测 1 个初始状态**（`num_trials_per_task: 1`）。LIBERO-Plus 是 `libero` 包的 drop-in 替换，因此 LoongForge 的 LIBERO runner 可原样复用；`benchmark.variant: plus` 仅额外生成按扰动类别聚合的汇总。

pi05 已在全部 4 个套件上验证通过（见 [验证](#验证)）；runner 与模型无关，任何已能在标准 LIBERO 上运行的模型都可以直接跑 LIBERO-Plus。

## ⚠️ 与标准 LIBERO 评测的差异

如果已经有可用的标准 LIBERO 评测，只需关注以下几点：

| 项目 | 标准 LIBERO | LIBERO-Plus |
|---|---|---|
| `benchmark.name` | `libero` | 仍写 `libero`；**不要**写成 `libero_plus`（该字段会被当作 suite 名） |
| `benchmark.variant` | 不写，默认 `standard` | **`plus`** —— 只有设置后才会生成 `libero_plus_summary.csv`；保持默认会静默丢失按扰动类别聚合的汇总 |
| `benchmark.episodes_per_task` | 通常 10–50 | **1**（官方协议：每个任务只有 1 个初始状态） |
| `benchmark.max_tasks` | 通常取子集 | **0**（每个套件约 2,400–2,600 个任务全跑） |
| Python 环境 | 标准 `libero` | 需要**独立环境**安装 LIBERO-Plus（包名相同，不能共存） |
| Assets | 仓库自带 | **必须下载 `assets.zip`（6.4 GB）** —— 仓库中没有 `assets/` |
| `LIBERO_CONFIG_PATH` | 标准 LIBERO 配置 | 指向 LIBERO-Plus 的 `config.yaml` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | 通常不需要 | torch ≥ 2.6 加载 `.pruned_init` 时**必需**（runner 会自动设置） |
| 输出 | `results.jsonl` / `summary.csv` / `suite_summary.csv` | 同上，**额外生成 `libero_plus_summary.csv`**（7 类扰动 + Total） |

其余部分——模型设置（`action_horizon`、`q99`、`empty_cameras`、norm stats）、runner、PayloadBuilder 与策略服务端——都与标准 LIBERO 完全一致。

## Step 0：权重与归一化统计

| 项目 | 来源 |
|---|---|
| 权重 | [lerobot/pi05_libero](https://huggingface.co/lerobot/pi05_libero)（`model.safetensors` + tokenizer） |
| Tokenizer | `google/paligemma-3b-pt-224` |
| **归一化统计** | `gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`（[HTTPS](https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json)） |

必须使用 checkpoint **自带**的 norm stats。将 openpi 的字段结构转换为 runner 期望的 LeRobot 键名：

```python
import json
src = json.load(open("norm_stats.json"))["norm_stats"]
out = {
    "observation.state": {k: src["state"][k] for k in ("mean", "std", "q01", "q99")},
    "action":            {k: src["actions"][k] for k in ("mean", "std", "q01", "q99")},
}
json.dump(out, open("pi05_libero_norm_stats.json", "w"), indent=2)
```

## Step 1：环境准备

只需按 [LIBERO 官方要求](https://github.com/Lifelong-Robot-Learning/LIBERO)安装环境（MuJoCo / robosuite / bddl 栈）。LIBERO-Plus 是 `libero` 包的 drop-in 替换，无需额外的仿真器配置——将其安装到独立的 benchmark 环境即可（标准 LIBERO、LIBERO-Plus、LIBERO-Pro 的包名相同，不能共存），并安装额外的客户端依赖：

```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git <LIBERO-PLUS>
pip install --no-build-isolation --config-settings editable_mode=compat -e <LIBERO-PLUS>
pip install websockets msgpack msgpack-numpy pyyaml wand "scikit-image==0.26.0" "usd-core>=25.5"
```

`wand` 与 `scikit-image` 属于 LIBERO-Plus 的 `extra_requirements`；`envs/env_wrapper.py` 会无条件导入它们。安装 scikit-image 时保持 `numpy==1.26.4`（较新版本可能拉入 NumPy 2.x，仿真栈不支持）。仓库顶层没有 `libero/__init__.py`，因此必须使用 `editable_mode=compat`（或在 site-packages 放一个指向源码根目录的 `.pth`），否则 `import libero` 会失败。离屏渲染还需要 EGL/OSMesa 与 ImageMagick 运行库：

```bash
apt-get install -y libegl1 libegl-mesa0 libosmesa6 libfontconfig1-dev libmagickwand-dev
```

### 1.1 本文结果所用的关键依赖版本

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| `libero`（LIBERO-Plus，editable） | 0.1.0 |
| torch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 |
| numpy | 1.26.4 |
| mujoco | 3.3.1 |
| robosuite | 1.4.0 |
| bddl | 1.0.1 |
| robomimic | 0.2.0 |
| transformers | 4.57.3 |
| opencv-python | 4.10.0.84 |
| scikit-image | 0.26.0 |
| wand | 0.7.2 |
| usd-core | 26.11 |
| websockets | 17.4 |
| msgpack / msgpack-numpy | 1.1.0 / 0.4.8 |
| PyYAML | 6.0.3 |
| imageio | 2.37.4 |
| h5py | 3.16.0 |
| apt（`libegl1` / `libegl-mesa0` / `libosmesa6` / `libmagickwand-dev`） | 1.7.0 / 25.2.8 / 25.1.7 / 8:6.9.12.98+dfsg1 |

> 以上为 benchmark/客户端侧版本。策略服务端运行在独立的 LoongForge 环境中。

### 1.2 下载并解压 assets（必需）

仓库**不包含** `assets/`；6.4 GB 的压缩包提供 10,030 个扰动任务所需的网格、纹理与场景。缺少它时环境创建会失败。

```bash
HF_HUB_DISABLE_XET=1 hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset --local-dir <ASSETS>
```

压缩包内部带有很深的目录前缀（`inspire/hdd/.../LIBERO-plus-0/assets/`），建议先解压到同一文件系统的临时目录，再把 `assets/` 移动到目标位置：

```bash
unzip -q <ASSETS>/assets.zip -d <ASSETS>/staging
mv <ASSETS>/staging/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets \
   <LIBERO-PLUS>/libero/libero/assets
```

期望结果：`<LIBERO-PLUS>/libero/libero/assets/{new_objects,textures,scenes,stable_scanned_objects,...}`。

### 1.3 LIBERO 路径配置

将 LIBERO 指向 LIBERO-Plus 的代码目录（不要复用标准 LIBERO 的配置）：

```yaml
# <PLUS-CONFIG>/config.yaml
benchmark_root: <LIBERO-PLUS>/libero/libero
bddl_files:    <LIBERO-PLUS>/libero/libero/bddl_files
init_states:   <LIBERO-PLUS>/libero/libero/init_files
datasets:      <LIBERO-PLUS>/libero/libero/../datasets
assets:        <LIBERO-PLUS>/libero/libero/assets
```

在评测 YAML 中把 `env.libero_config_path` 指向包含该文件的目录。

⚠️ runner 需要加载 `.pruned_init` pickle（内部是 `numpy` 数据），而 `torch >= 2.6` 默认的 `weights_only=True` 会拒绝加载。runner 自身会设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；单独运行 runner 时也需要手动导出该变量。

## Step 2：运行评测

启动脚本会运行 YAML 中 `sweep.benchmark.suite` 列出的套件（随仓库提供的模板会跑全部 4 个套件）：

```bash
cd /path/to/LoongForge
BENCHMARK_PYTHON=/path/to/libero_plus/bin/python \
  bash examples/embodied/pi05/eval/run_libero_plus_eval.sh
```

环境变量：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `CONFIG` | 评测 YAML（也可作为第一个位置参数传入） | `examples/embodied/pi05/eval/configs/libero/libero_plus.yaml` |
| `BENCHMARK_PYTHON` | LIBERO-Plus 环境的解释器（客户端） | `/path/to/libero_plus/bin/python` |
| `CUDA_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID` | 客户端与策略服务使用的 GPU | `0` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | 客户端保持离线 | `1` |
| `DRY_RUN` | 设为 `1` 时只打印环境变量与命令 | `0` |

关键配置字段（`configs/libero/libero_plus.yaml`）：

- `benchmark.variant: plus` —— 启用 `libero_plus_summary.csv`；标准 LIBERO 不需要
- `benchmark.suite` / `sweep.benchmark.suite` —— 单个套件 / 全部 4 个套件
- `benchmark.max_tasks: 0`、`benchmark.episodes_per_task: 1` —— 全部任务，官方单次试验协议
- `benchmark.max_steps: -1` —— 使用套件默认步数（Spatial 220 / Object 280 / Goal 300 / Long 520）
- `benchmark.continuous_gripper: true`、`prefer_controller_ee: false` —— pi05 在 LIBERO 上的约定
- `model.action_horizon: 10` —— **必须与 checkpoint 一致**（openpi `pi05_libero` 使用 10；设为 50 会导致成功率崩塌）
- `model.eval_image_normalize_mode: siglip`、`model.eval_normalization_mode: q99` —— 与 openpi `LiberoInputs` 管线一致
- `model.state_encoding: libero_ee8` —— 8 维本体状态（位置 + axis-angle + 夹爪 qpos）
- `model.empty_cameras: 1` —— 官方 checkpoint 期望 3 个视觉槽（base + wrist + 1 个被 mask 的填充槽）；2 相机 checkpoint 设为 `0`
- `server.chunk_execute_steps: 10` —— 每个预测 chunk 执行 10 步后重新规划
- `server.dataset_statistics_path` —— Step 0 中 checkpoint 自带的 norm stats（不要使用数据集派生的 stats 文件）

每个套件的输出：`results.jsonl`、`summary.csv`、`suite_summary.csv`，以及 **`libero_plus_summary.csv`**（7 类扰动 + 套件 `Total` + `all/Total`）。

## 验证

pi05（`lerobot/pi05_libero`），每个套件全部任务，每任务 1 episode，seed 7，**10,030 episodes，0 次基础设施失败**。参考值为 [ELAN4D](https://arxiv.org/pdf/2605.30484) 论文 Table 1 中的 π0.5 行。

表格同时合并了两个维度：**扰动**行是 4 个套件的无权重平均，**套件**行按套件统计，**Overall** 为全部 10,030 个任务的汇总。

| 维度 | 指标 | LoongForge | 论文 π0.5 | Δ |
|---|---|---|---|---|
| 扰动 | 相机视角 | 57.4% | 59.7% | −2.3 |
| 扰动 | 机器人初始状态 | 82.6% | 65.5% | +17.1 |
| 扰动 | 指令语义 | 83.7% | 75.3% | +8.4 |
| 扰动 | 光照条件 | 93.5% | 87.0% | +6.5 |
| 扰动 | 背景纹理 | 93.2% | 82.4% | +10.8 |
| 扰动 | 传感器噪声 | 84.5% | 72.1% | +12.4 |
| 扰动 | 物体布局 | 86.4% | 80.3% | +6.1 |
| 任务套件 | Spatial | 89.1% | 79.9% | +9.2 |
| 任务套件 | Object | 87.1% | 87.8% | −0.7 |
| 任务套件 | Goal | 75.4% | 69.0% | +6.4 |
| 任务套件 | Long (10) | 75.6% | 64.9% | +10.7 |
| Overall | 全部任务（汇总） | 81.7% | 73.6% | +8.1 |
