# AoE-Repro —— AoE 云端标注流水线的 Mac 可运行复现

对论文 *AoE: Always-on Egocentric Human Video Collection for Embodied AI*
（[arXiv 2602.23893](https://arxiv.org/abs/2602.23893)）中**云端六阶段标注流水线**的
忠实再实现，可在 **Apple Silicon（MPS/CPU，无需 CUDA）** 上运行。它把一段第一人称
（egocentric）视频转成可训练标注 —— **3D 手部关节（世界坐标）、6 自由度相机位姿、单目深度、
原子动作分段** —— 每个重型模型都替换成可在 Mac 上跑的轻量开源替身。

📖 English: [README.md](README.md)

> **范围。** 原论文是一套*数据采集 + 数据治理系统*（颈挂硬件 + 手机 App + 云端流水线 +
> 机器人策略验证），且**未发布任何代码、数据或模型**。本仓库只复现**云端标注流水线**。
> 硬件支架、iOS/Android App、Unitree-G1 下游机器人策略需要实体设备，不在范围内。

## 流水线：论文模型 → Mac 替身

| # | 阶段 | 论文（重型） | 本仓库（Mac 可运行） |
|---|------|--------------|----------------------|
| 1 | 标定 | Camera2 内参 | 元数据 → OpenCV 棋盘格 → FOV 针孔默认 |
| 2 | 动作分段 | Qwen3-VL-235B | 光流 + 手在场启发式；可插拔 VLM 后端 |
| 3 | 相机轨迹 + 深度 | MegaSAM + Lingbot-Depth | OpenCV ORB 单目里程计 + Depth-Anything-V2-Small（MPS） |
| 4 | 手部重建 | HaWoR + MANO | MediaPipe 21 关节 → 深度反投影 → 世界系 → 平滑 |
| 5 | 数据增强（可选） | Masquerade GAN + 视频扩散 | 人像分割换背景 + `cv2.inpaint`（默认关闭） |
| 6 | 质检 | 3σ 速度 + 5px 重投影 | NumPy 速度/重投影过滤 + 5% 人工抽检采样 |
| + | 评测 | — | MPJPE / PA-MPJPE / AUC(PCK)；ATE / ATE-S / RPE（用 `evo`） |

执行 DAG：`ingest → trajectory → hands → segment → qc`（augment 可选）。

## 安装

需要 Python 3.11/3.12（MediaPipe/Depth-Anything 的 wheel 落后于最新 Python）与
[`uv`](https://docs.astral.sh/uv/)。

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .          # 核心流水线
# 可选 extras：
uv pip install -e '.[vlm]'       # 托管 VLM 分段后端
uv pip install -e '.[augment]'   # 重型扩散增强路径
uv pip install -e '.[download]'  # 数据集下载脚本
```

首次运行会自动下载两个小模型：Depth-Anything-V2-Small（HF 缓存）和
`hand_landmarker.task`（`~/.cache/aoe_pipeline/`）。

## 快速开始

```bash
# 1. 自包含合成视频（无手）：跑通 depth/VO/segment/QC
python scripts/make_sample_clip.py                       # -> data/sample_clip.mp4
aoe-pipeline run --video data/sample_clip.mp4 --output-dir output --verbose

# 2. 真手 demo：把一张手部照片做成第一人称风格视频
curl -sL -o datasets/hand.jpg \
  https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg
python scripts/make_hand_clip.py --image datasets/hand.jpg --out datasets/hand_clip.mp4
aoe-pipeline run --video datasets/hand_clip.mp4 --output-dir output --verbose
```

每次运行在 `output/<clip_id>/` 写出：帧、深度、`trajectory.tum`、
`hands/joints_world.npy`、`segments.json`、`qc_report.json`、`manifest.json`、`viz/`。

## 评测

```bash
# 相机轨迹 vs GT（Sim3 / SE3 / RPE）
aoe-pipeline eval-traj --est output/<clip>/trajectory.tum --gt <gt>.tum
# 手部姿态 vs GT（MPJPE / PA-MPJPE / AUC）
aoe-pipeline eval-hands --pred output/<clip>/hands/joints_world.npy --gt <gt>.npy --to-mano
```

GT 推荐用 **EgoDex**（Apple，自带 3D 手部关节）：`python scripts/download_egodex.py`
打印获取步骤；`scripts/download_ego4d.py` 覆盖需许可的 Ego4D 流程。

## 可视化

```bash
# 分段时间轴 / 拼图 / 标注视频 + 切出交互小片
python scripts/visualize_segments.py --clip-dir output/<clip> --source <video>.mp4

# HaWoR 式 正/俯/侧 正交三视图动画
python scripts/render_hand_views.py --clip-dir output/<clip> --frame both

# HaWoR 式 透视「世界场景」：地面 + 手部残影轨迹 + 相机路径 + 视锥（两视角）
python scripts/render_hand_views.py --clip-dir output/<clip> --frame scene
```

## 测试

```bash
pytest -q                 # 快速单元测试（schema、QC 数学、metrics、ingest）
AOE_RUN_E2E=1 pytest -q    # + 在合成视频上跑完整真实模型流水线
```

## 保真度说明

这是一个可运行的**原型**，**不是**论文精度复现：

- **轨迹是 up-to-scale 的**：单目 ORB 里程计只能恢复到尺度，平移按中位深度的比例缩放；
  评测请用 Sim(3)（7-DoF）与无尺度 SE(3) 对齐（`eval-traj` 都提供）。
- **深度是相对的**，非度量（Depth-Anything-V2 输出仿射不变逆深度，我们映射到伪米范围）。
- **手部重建**用「21 点检测 + 单目深度」替代 HaWoR 的 MANO 网格拟合；因此 QC 的重投影项
  主要反映抖动检测与平滑估计的分歧（并非网格拟合质量），**速度过滤器**才是主力时序检查。
- **数据增强**是「分割 + inpaint」的轻量替身，替代论文的 GAN/扩散管线。

每个阶段都暴露与其重型对应物相同的接口，所以在 GPU 机器上换成真实模型（MegaSAM、HaWoR、
235B VLM）只是改配置/注册表，不用重写。

## 未复现部分

颈挂硬件、设备端 App + 选择性录制、下游 GR00T-N1.5 / Unitree-G1 机器人策略 —— 均需实体设备。

## 免责声明与数据

- 本项目为**独立复现**，用于研究/学习，**与 AoE 论文作者无隶属或背书关系**，使用轻量替身
  模型再实现其架构，并非论文精度或官方发布。
- 第三方模型（MediaPipe、Depth-Anything-V2 等）在运行时按其各自许可证下载，**不在此再分发**。
- **不提交任何个人或第三方媒体**：真实录制视频与所有生成产物（`output/`、`datasets/`）均被
  git 忽略；仅打包**合成的** `data/sample_clip.mp4` 以便开箱即用，真手 demo 请自行用
  `scripts/make_hand_clip.py` 生成。
- 代码以 [MIT 许可证](LICENSE) 发布。
