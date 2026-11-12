# LIBERO-Pro 评测

[LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO)（[论文](https://arxiv.org/abs/2510.03827)）是 LIBERO 的鲁棒性扩展基准。它保留原有的 4 个套件（Spatial、Object、Goal、Long/10），并为每个套件增加 5 个扰动维度：

| 官方列 | 扰动类型 | 注册套件名 |
|---|---|---|
| **Obj** | 物体外观 / 颜色 / 尺寸 | `libero_<base>_object` |
| **Pos** | 物体初始位置 | `libero_<base>_swap` |
| **Sem** | 指令语义改写 | `libero_<base>_lan` |
| **Task** | 任务逻辑 / 目标重设计 | `libero_<base>_task` |
| **Env** | 工作环境替换 | `libero_<base>_env` |

即 **20 个套件 × 10 个任务**，每个任务评测 **50 个 episode**（每个 `.pruned_init` 含 50 个初始状态）。LIBERO-Pro 是 `libero` 包的 drop-in 替换，因此 LoongForge 的 LIBERO runner 可原样复用。

pi05 已在全部 5 个维度上验证通过（见 [验证](#验证)）。

## ⚠️ 与标准 LIBERO 评测的差异

如果已经有可用的标准 LIBERO 评测，只需关注以下几点：

| 项目 | 标准 LIBERO | LIBERO-Pro |
|---|---|---|
| `benchmark.name` | `libero` | 仍写 `libero`；**不要**写成 `libero_pro`（该字段会被当作 suite 名） |
| `benchmark.variant` | 不写，默认 `standard` | **`pro`** —— 只有设置后才会从 BDDL 读取指令；保持默认时 `_lan` / `_task` 会退回文件名推导的指令 |
| `benchmark.suite` / `sweep.benchmark.suite` | `libero_spatial` / `object` / `goal` / `10` | 派生名 `libero_<base>_{object,swap,lan,task,env}` —— **20 个套件** |
| `benchmark.episodes_per_task` | 通常 10–50 | **50**（官方协议） |
| `benchmark.max_tasks` | 通常取子集 | **0**（每个套件的 10 个任务全跑） |
| Python 环境 | 标准 `libero` | 需要**独立环境**安装 LIBERO-Pro（包名相同，不能共存） |
| 数据 | 4 个标准套件 | HF 数据集（约 10 MB）提供 Obj/Pos/Sem/Task **+ 本地生成 `_env` 套件**（`GEN_ENV=1`） |
| `LIBERO_CONFIG_PATH` | 标准 LIBERO 配置 | 指向 LIBERO-Pro 的 `config.yaml` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | 通常不需要 | torch ≥ 2.6 加载 `.pruned_init` 时**必需**（runner 会自动设置） |
| 指令来源 | 任务文件名 | `_lan` / `_task` 使用 **BDDL `(:language ...)`**（所以需要 `variant: pro`） |
| 输出 | `results.jsonl` / `summary.csv` / `suite_summary.csv` | 同上，每个套件一个目录（`sweep_XXX/`） |

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

只需按 [LIBERO 官方要求](https://github.com/Lifelong-Robot-Learning/LIBERO)安装环境（MuJoCo / robosuite / bddl 栈）。LIBERO-Pro 是 `libero` 包的 drop-in 替换，无需额外的仿真器配置——将其安装到独立的 benchmark 环境即可（标准 LIBERO、LIBERO-Plus、LIBERO-Pro 的包名相同，不能共存）：

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git <LIBERO-PRO>
pip install --no-build-isolation --config-settings editable_mode=compat -e <LIBERO-PRO>
pip install websockets msgpack msgpack-numpy pyyaml
```

仓库顶层没有 `libero/__init__.py`，因此必须使用 `editable_mode=compat`（或在 site-packages 放一个指向源码根目录的 `.pth`），否则 `import libero` 会失败。LIBERO-Pro 自带 LIBERO 的 assets，无需额外下载。

### 1.1 本文结果所用的关键依赖版本

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| `libero`（LIBERO-Pro，editable） | 0.1.0 |
| torch / torchvision | 2.9.0 / 0.24.0 |
| numpy | 1.26.4 |
| mujoco | 3.3.1 |
| robosuite | 1.4.0 |
| bddl | 1.0.1 |
| robomimic | 0.2.0 |
| transformers | 4.57.3 |
| opencv-python | 4.10.0.84 |
| websockets | 17.1 |
| msgpack / msgpack-numpy | 1.1.0 / 0.4.8 |
| PyYAML | 6.0.3 |
| imageio | 2.37.4 |
| h5py | 3.16.0 |
| scipy | 1.17.1 |

> 以上为 benchmark/客户端侧版本。策略服务端运行在独立的 LoongForge 环境中。

### 1.2 加载官方预生成的 4 个维度

HF 数据集 [zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)（约 10 MB）提供 **Obj / Pos / Sem / Task** 四个维度的 `bddl_files/<suite>/` 与 `init_files/<suite>/`：

```bash
hf download zhouxueyang/LIBERO-Pro --repo-type dataset --local-dir <PRO-DATA>
for base in libero_spatial libero_object libero_goal libero_10; do
  for dim in lan object swap task; do
    cp -r <PRO-DATA>/bddl_files/${base}_${dim} <LIBERO-PRO>/libero/libero/bddl_files/
    cp -r <PRO-DATA>/init_files/${base}_${dim} <LIBERO-PRO>/libero/libero/init_files/
  done
done
```

### 1.3 生成 Env 维度

`libero_*_env` 的 bddl/init 由官方管线在评测时现场生成。**启动脚本 `examples/embodied/pi05/eval/run_libero_pro_eval.sh` 已内置该生成逻辑**：设置 `GEN_ENV=1` 运行即可，脚本会在评测前生成缺失的 `libero_*_env` 套件（需要 `LIBERO_PRO_ROOT`；若某个 suite 的 `*_env` 目录已存在且非空则跳过）。也可以用同一套官方管线手动生成一次：

```python
import os, shutil, sys, yaml
root = "<LIBERO-PRO>"; base = f"{root}/libero/libero"; sys.path.insert(0, root)
import perturbation

cfg = yaml.safe_load(open(f"{root}/evaluation_config.yaml"))
ood = {k: os.path.join(root, v.lstrip("./")) for k, v in cfg["ood_task_configs"].items()}
for suite in ("libero_goal", "libero_spatial", "libero_10", "libero_object"):
    tmp = perturbation.process_bddl_file_mixed(
        input_dir=f"{base}/bddl_files/{suite}", task_suite_name=suite,
        flags=perturbation.PerturbFlags(use_environment=True), configs=ood, seed=42)
    perturbation.EvalEnvCreator(
        input_dir=tmp, script_path=f"{root}/notebooks/generate_init_states.py",
        base_output_dir=f"{base}/init_files").create_env()
    for kind in ("bddl_files", "init_files"):
        shutil.rmtree(f"{base}/{kind}/{suite}_env", ignore_errors=True)
        os.rename(f"{base}/{kind}/{suite}_temp", f"{base}/{kind}/{suite}_env")
```

### 1.4 LIBERO 路径配置

将 LIBERO 指向 LIBERO-Pro 的代码目录（不要复用标准 LIBERO 的配置）：

```yaml
# <PRO-CONFIG>/config.yaml
benchmark_root: <LIBERO-PRO>/libero/libero
bddl_files:    <LIBERO-PRO>/libero/libero/bddl_files
init_states:   <LIBERO-PRO>/libero/libero/init_files
datasets:      <LIBERO-PRO>/libero/libero/../datasets
assets:        <LIBERO-PRO>/libero/libero/assets
```

在评测 YAML 中把 `env.libero_config_path` 指向包含该文件的目录。

⚠️ runner 需要加载 `.pruned_init` pickle（内部是 `numpy` 数据），而 `torch >= 2.6` 默认的 `weights_only=True` 会拒绝加载。runner 自身会设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；单独运行 runner 时也需要手动导出该变量。

## Step 2：运行评测

启动脚本会运行 YAML 中 `sweep.benchmark.suite` 列出的**全部 20 个套件**：

```bash
cd /path/to/LoongForge
BENCHMARK_PYTHON=/path/to/libero_pro/bin/python \
LIBERO_PRO_ROOT=<LIBERO-PRO> \
  bash examples/embodied/pi05/eval/run_libero_pro_eval.sh
```

环境变量：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `CONFIG` | 评测 YAML | `examples/embodied/pi05/eval/configs/libero/libero_pro.yaml` |
| `BENCHMARK_PYTHON` | LIBERO-Pro 环境的解释器（客户端） | `/path/to/libero_pro/bin/python` |
| `LIBERO_PRO_ROOT` | LIBERO-Pro 代码目录（用于预检） | 未设置 |
| `GEN_ENV` | 设为 `1` 时先生成缺失的 `libero_*_env` 套件 | `0` |
| `CUDA_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID` | 客户端与策略服务使用的 GPU | `0` |
| `DRY_RUN` | 设为 `1` 时只打印预检结果与命令 | `0` |

关键配置字段（`configs/libero/libero_pro.yaml`）：

- `benchmark.variant: pro` —— 从 BDDL 的 `(:language ...)` 段读取指令，而不是用文件名推导。默认值为 `standard`，因此**必须显式设置**；否则 `_lan` / `_task` 会退回到文件名推导的指令。
- `benchmark.suite` / `sweep.benchmark.suite` —— 单套件兜底值 / 完整 20 套件矩阵
- `benchmark.episodes_per_task: 50` —— 官方协议；`max_tasks: 0` 表示全部 10 个任务
- `benchmark.max_steps: -1` —— 使用基础套件步数（Spatial 220 / Object 280 / Goal 300 / Long 520）
- `benchmark.continuous_gripper: true`、`prefer_controller_ee: false` —— pi05 在 LIBERO 上的约定
- `model.action_horizon: 10` —— **必须与 checkpoint 一致**（openpi `pi05_libero` 使用 10）
- `model.eval_image_normalize_mode: siglip`、`model.eval_normalization_mode: q99` —— 与 openpi `LiberoInputs` 管线一致
- `model.state_encoding: libero_ee8` —— 8 维本体状态（位置 + axis-angle + 夹爪 qpos）
- `model.empty_cameras: 1` —— 官方 checkpoint 期望 3 个视觉槽（base + wrist + 1 个被 mask 的填充槽）；2 相机 checkpoint 设为 `0`
- `server.chunk_execute_steps: 10` —— 每个预测 chunk 执行 10 步后重新规划
- `server.dataset_statistics_path` —— Step 0 中 checkpoint 自带的 norm stats（不要使用数据集派生的 stats 文件）

## 验证

pi05（`lerobot/pi05_libero`），每任务 50 episodes，seed 7，**10,000 episodes，0 次基础设施失败**。每个单元格为 **LoongForge / 官方 π0.5** 的成功率（%），参考值来自 [LIBERO-Pro 榜单](https://github.com/Zxy-MLlab/LIBERO-PRO#-libero-pro-model-leaderboard)：

| suite | Obj | Pos | Sem | Task | Env | avg |
|---|---|---|---|---|---|---|
| Spatial | **97.4** / 97 | **50.4** / 20 | **95.6** / 97 | **45.8** / 1 | **53.6** / 46 | **68.6** / 52.2 |
| Object | **94.2** / 98 | **26.6** / 17 | **99.4** / 96 | **10.2** / 1 | **75.2** / 73 | **61.1** / 57.0 |
| Goal | **74.8** / 97 | **27.6** / 38 | **95.0** / 97 | **21.2** / 0 | **31.2** / 46 | **50.0** / 55.6 |
| Long (10) | **65.2** / 92 | **10.0** / 8 | **93.8** / 93 | **17.6** / 1 | **65.8** / 46 | **50.5** / 48.0 |
| **Overall** | | | | | | **57.5** / **53.2** |

> 官方每套件的 `avg` 为其 5 列的算术平均；榜单公布的 `Total` 为 **0.53**，与这里的 53.2% 一致。