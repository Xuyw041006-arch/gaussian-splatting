# Semantic-Adaptive 3D Gaussian Splatting

[![Open L4/T4 high-quality training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xuyw041006-arch/gaussian-splatting/blob/main/colab_t4_full_smoke_test.ipynb)
[![Open Ramen joint benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xuyw041006-arch/gaussian-splatting/blob/main/colab_ramen_joint_benchmark.ipynb)

这是基于 Graphdeco 官方 `gaussian-splatting` 主分支的可运行扩展。RGB 训练、CUDA 光栅化、COLMAP 数据读取、深度正则化和 SIBR Viewer 都来自原始 3DGS；本仓库新增训练时联合语义优化、三级重要性资源分配、开放词汇搜索、非破坏性删除和点击检查。跨视图原型与多粒度门控分别受 LaGa、SAGA 启发，但不是两篇论文代码的逐行复现。

> 使用范围继承上游 [LICENSE.md](LICENSE.md)：仅限非商业研究与评估。

## 已实现的闭环

1. **自建数据**：普通照片复制到标准目录，COLMAP 自动计算相机与稀疏点云。
2. **重要物体优先**：用户或 LLM 给出 `apple,cup` 等重要词；SAM+CLIP 生成重要、普通、背景三级监督，联合训练提高重点区域的 RGB/语义损失和高斯密度。
3. **开放词汇语义**：SAM 区域按 predicted-IoU × stability 加权，ViT-H/14 CLIP 特征经 PCA 压缩后与 RGB 同时反向传播；alpha 归一化、场景原型和动态 3D 邻域损失共同抑制跨视图漂移。
4. **少图模式**：可均匀限制训练视角，并复用官方主分支的单目深度正则化；对视频帧支持 sequential COLMAP matcher。
5. **搜索和编辑**：文本查询返回全部匹配高斯、数量、中心和包围盒；可导出命中点云，或生成删除/仅保留目标的新模型。
6. **网页交互**：[Gaussian Atlas](https://gaussian-atlas-xyw.xuyw041006.chatgpt.site) 可直接导入训练结果，在浏览器中拖动旋转、滚轮缩放、点击拾取、文本定位并可恢复地隐藏语义物体。

## 目录

```text
preprocess_semantics.py       SAM + CLIP + PCA + 三级层级监督
train.py                      官方 3DGS + 联合语义/重要性训练
train_semantics.py            语义特征蒸馏到固定 3D Gaussians
semantic_query.py             “找到所有苹果”
semantic_edit.py              删除/仅保留查询结果，源模型不改
gaussian_transform.py         移动/旋转/缩放全部或语义选中高斯
semantic_inspect.py           指定视图与像素，返回点和语义信息
semantic_viewer.py            文本搜索 + 鼠标点击 Web UI
export_web_bundle.py          导出 Gaussian Atlas 的 PLY + 语义索引
scripts/prepare_dataset.py    自建图片目录导入
scripts/prune_gaussians.py    尺度/透明度/空间离群高斯清理
scripts/run_pipeline.py       完整流水线与 dry-run
scripts/run_ramen_benchmark.py Ramen 顺序基线/联合模型 A/B 测试
scripts/evaluate_lerf_mask.py  PSNR/SSIM/mIoU/Boundary-IoU
scripts/preflight.py          环境/数据/检查点诊断
semantic/                     查询、投影和拾取的公共逻辑
tests/                        无 GPU 单元与命令规划测试
```

## 1. 安装

要求 Linux、NVIDIA GPU、CUDA；建议显存 12 GB 以上。先递归克隆，再按官方环境安装：

```bash
git clone --recursive https://github.com/Xuyw041006-arch/gaussian-splatting.git
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
pip install -r requirements-semantic.txt
pip install -r requirements-ui.txt
```

下载与 `--sam_model` 一致的 Segment Anything 检查点。快速测试可用 `sam_vit_b_01ec64.pth`；高质量模式使用 `sam_vit_h_4b8939.pth`。

## 2. 自建数据

拍摄建议：围绕场景移动、相邻图像保持 60% 以上重叠、避免纯旋转和运动模糊；少图模式建议至少 8 张。

```bash
python scripts/prepare_dataset.py \
  --images /data/my_photos \
  --scene /data/my_scene
```

视频抽帧请先用 ffmpeg，再把抽帧目录传给上面的命令。顺序视频帧在流水线中使用 `--matcher sequential`。

## 3. 先检查，不训练

```bash
python scripts/preflight.py \
  --scene /data/my_scene \
  --sam_checkpoint /checkpoints/sam_vit_b_01ec64.pth

python scripts/run_pipeline.py \
  --scene /data/my_scene \
  --model output/my_scene \
  --sam_checkpoint /checkpoints/sam_vit_b_01ec64.pth \
  --important "apple,cup" \
  --sparse --max_train_views 8 \
  --dry_run
```

`--dry_run` 只打印四个真实命令，不生成模型。

## 4. L4/T4 多视角高质量训练与冒烟模式

没有本地 NVIDIA GPU 时，可直接打开上方 Colab Notebook，选择 **L4 GPU（推荐）** 或成本更低的 T4。默认使用 NeRF Synthetic Lego 的 80 个训练视角、10 个验证视角、512 像素分辨率和 10 万初始化点，执行 40,000 次 RGB 与 8,000 次语义训练。RGB 阶段保留上游 3DGS 的梯度累计、clone/split 和 opacity pruning，把分裂延长到 22,000 轮；语义阶段使用 SAM ViT-H、OpenCLIP ViT-H/14、24 维场景特征、alpha 归一化和颜色感知 3D KNN 正则。Notebook 会备份原始 PLY，高细节清理预设预计保留约 17 万高斯，而不是此前约 6.8 万的体积优先版本。最后生成可导入 [Gaussian Atlas](https://gaussian-atlas-xyw.xuyw041006.chatgpt.site) 的网页包。Lego 阈值只适用于该演示场景；自建数据应根据尺度重新调节。

若只想先验证环境，可在命令行使用下面的小迭代冒烟模式；它只验证 COLMAP、官方 3DGS rasterizer、语义预处理和语义蒸馏能够完整走通，不代表重建质量：

```bash
python scripts/run_pipeline.py \
  --scene /data/my_scene \
  --model output/my_scene_smoke \
  --sam_checkpoint /checkpoints/sam_vit_b_01ec64.pth \
  --important "apple,cup" \
  --scene_iterations 100 \
  --semantic_iterations 20 \
  --feature_width 160 \
  --sparse --max_train_views 8
```

成功标准是同时存在：

```text
output/my_scene_smoke/point_cloud/iteration_100/point_cloud.ply
output/my_scene_smoke/semantic/iteration_100/semantic_features.pt
```

正式训练去掉两个小迭代参数，默认 RGB 40,000 次、语义 8,000 次：

```bash
python scripts/run_pipeline.py \
  --scene /data/my_scene \
  --model output/my_scene \
  --sam_checkpoint /checkpoints/sam_vit_b_01ec64.pth \
  --important "apple,cup" \
  --sparse --max_train_views 8 \
  --resume
```

### 单目深度增强少图重建

官方主分支已原生支持 Depth Anything V2 深度正则。按下方上游说明生成 `depths/` 和 `sparse/0/depth_params.json` 后，在流水线增加：

```bash
--depths depths
```

这会真正把深度损失加入训练，而不是仅修改“稀疏”参数名称。

### 接入 LLM 的逐图重点物体

让任意 LLM 输出一个 JSON 文件即可，不绑定某个云端模型：

```json
{
  "IMG_0012.jpg": ["apple", "red cup"],
  "IMG_0013.jpg": ["apple"],
  "IMG_0014": ["fruit bowl", "table"]
}
```

然后把 `--important "apple,cup"` 换为 `--important_json important_objects.json`。脚本会为每张图分别生成重要区域；JSON 没覆盖的图片使用 `--important` 的全局词表（如同时提供）。

## 5. 搜索、删除与点击

找到场景内所有苹果，并导出命中点云：

```bash
python semantic_query.py \
  --model output/my_scene \
  --text "apple" \
  --threshold 0.25 \
  --output output/apples.npz \
  --export_selected output/apples.ply \
  --device cpu
```

删除苹果会创建新模型，绝不覆盖源模型：

```bash
python semantic_edit.py \
  --model output/my_scene \
  --selection output/apples.npz \
  --output_model output/my_scene_without_apples \
  --action remove
```

移动、旋转或缩放刚才找到的苹果，同样输出为新模型：

```bash
python gaussian_transform.py \
  --model output/my_scene \
  --selection output/apples.npz \
  --output_model output/my_scene_moved_apples \
  --translate 0.2 0 0 --rotate_z 30 --scale 1.2
```

在某张注册图像的 `(x, y)` 像素检查对象，并用候选词解释：

```bash
python semantic_inspect.py \
  --model output/my_scene \
  --view IMG_0012.jpg --x 640 --y 420 \
  --labels "apple,cup,table"
```

导出独立网页所需的模型与语义索引：

```bash
python export_web_bundle.py \
  --model output/my_scene \
  --labels "apple,cup,table" \
  --threshold 0.25 \
  --output_dir output/my_scene_web
```

打开 [Gaussian Atlas](https://gaussian-atlas-xyw.xuyw041006.chatgpt.site)，先导入 `output/my_scene_web/point_cloud.ply`，再导入 `semantic_objects.json`。模型只在浏览器本地读取，不会上传；网页中的隐藏操作也不覆盖源文件。

旧的已注册视图 UI 仍可本地启动：

```bash
python semantic_viewer.py \
  --model output/my_scene \
  --source /data/my_scene \
  --device cpu
```

浏览器打开 `http://127.0.0.1:7860`。自由旋转、平移和缩放完整高斯场景仍使用官方 SIBR Viewer：

```bash
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/my_scene
```

若网页视图中出现少量巨大雾状高斯或远处散点，先写入新 PLY 进行可恢复清理：

```bash
python scripts/prune_gaussians.py \
  --input output/my_scene/point_cloud/iteration_40000/point_cloud.ply \
  --output output/my_scene/point_cloud/iteration_40000/point_cloud.clean.ply \
  --max_scale 0.014 --min_opacity 0.005 --max_radius 1.30
```

这里是“高细节”而非“最小文件”预设。`max_scale` 和 `max_radius` 与场景尺度相关，不应盲目复制到自建数据；请同时比较高斯数量、保留视角的 PSNR/SSIM/LPIPS 和实际画面。

## 6. 无 GPU 的代码测试

这些测试验证稀疏视角选择、语义数值变换、点击投影、流水线参数和源码可编译性：

```bash
python -m unittest discover -s tests -v
python -m compileall -q arguments gaussian_renderer scene semantic scripts \
  train.py train_semantics.py preprocess_semantics.py \
  semantic_query.py semantic_edit.py semantic_inspect.py semantic_viewer.py \
  export_web_bundle.py scripts/evaluate_lerf_mask.py \
  scripts/run_ramen_benchmark.py \
  gaussian_transform.py
```

无 GPU 测试不能证明 CUDA 扩展或训练可用；必须再执行上面的 GPU 冒烟测试。

## 7. 联合重建与三级语义（LaGa + SAGA inspired）

默认流水线现已改为联合训练：RGB、层级语义和重要性在同一个 `train.py`
反向传播中优化。语义梯度与 RGB 梯度共同进入原版 3DGS 的
`clone/split/prune`，不再等到重建结束后才附加语义。

| 档位 | RGB 权重 | 语义权重 | 分裂阈值倍率 | opacity 剪枝倍率 | 有效 SH 阶数 |
|---|---:|---:|---:|---:|---:|
| 背景 | 0.35 | 0.15 | 1.80 | 2.00 | 1 |
| 普通物品 | 1.00 | 1.00 | 1.00 | 1.00 | 3 |
| 重要物品 | 4.00 | 4.00 | 0.55 | 0.50 | 5 |

- **LaGa-inspired 跨视图一致性**：将全场景 SAM 区域的 CLIP 描述符聚成
  64 个原型，再按区域置信度、簇内紧致度和描述符相似度以 0.65 权重聚合。
- **SAGA-inspired 多粒度**：SAM 区域按画面面积分为 coarse（≥25%）、
  middle（5%–25%）和 fine（<5%），训练时轮换监督并学习尺度通道门控。
- **动态资源分配**：重要高斯更容易分裂、更难被删除，并允许五阶 SH；背景
  高斯更难分裂、会更早剪枝，只训练一阶 SH。当前光栅器仍以全局最大 SH 阶数
  执行，因此实际算力节省主要来自高斯数量分配。

用户或 LLM 通过 `--important` / `--important_json` 提供重要物品，通过
`--normal` / `--normal_json` 提供普通物品，未匹配区域作为背景。

```bash
python scripts/run_pipeline.py \
  --scene data/my_scene --model output/my_scene \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --important "egg,pork belly,wavy noodles in bowl" \
  --normal "yellow bowl,chopsticks,glass of water"
```

使用 `--training_mode sequential` 可保留旧的“先 RGB、后语义”基线。

### LERF-Mask ramen 对比

下面的脚本使用官方 `test_*.jpg` 与 `test_mask/0..2` 切分，对仓库原有
“先重建、后语义”的顺序基线和新联合模型计算 PSNR、SSIM、mIoU 与
Boundary-IoU：

```bash
python scripts/run_ramen_benchmark.py \
  --scene data/lerf_mask/ramen \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --output_root output/ramen_ablation
```

已在 Colab NVIDIA L4 上完成 1,500 轮 pilot（3 个有标注测试视图，固定
semantic threshold 0.25）：

| 指标 | 顺序基线 | 联合模型 | 联合 - 基线 |
|---|---:|---:|---:|
| 高斯数量 | 94,708 | 52,553 | -44.5% |
| 全局 PSNR | 26.27 dB | 25.28 dB | -0.99 dB |
| 重要区域 PSNR | 28.83 dB | 28.61 dB | -0.23 dB |
| 全局 mIoU | 15.88% | 14.72% | -1.16 pp |
| 重要类别 mIoU | 10.49% | 12.21% | **+1.72 pp** |
| Boundary-IoU | 7.21% | 4.00% | -3.21 pp |

联合模型的 22.89% 重要监督像素获得了 44.58% 的最终高斯，证明三级资源分配
已经工作；但短训练下全局画质和边界质量仍落后，不能宣称全面优于现有方法。
原始指标见 [`benchmarks/ramen_pilot_1500.json`](benchmarks/ramen_pilot_1500.json)，
完整公式、阈值、限制和下一轮优化见 [`docs/JOINT_MODEL.md`](docs/JOINT_MODEL.md)。
Notebook 的 full 模式默认运行 15,000/5,000 轮；联合语义从第 1,000 轮开启，
在第 10,000 轮结束 densification，并保留最后 5,000 轮稳定新分裂的高斯。
需要复现实验上限时仍可显式传入 `--iterations 30000`，其 densification 截止
保持第 15,000 轮。
Notebook 默认将公开数据、SAM 检查点、语义预处理和全部训练结果写入
`MyDrive/semantic_adaptive_3dgs/ramen_full_15k/`。加上 `--resume` 后，RGB/联合
训练从最近的 7k 或 10k 检查点继续，顺序语义训练每 1,000 轮保存 Adam 状态；
浏览器或 Colab 运行时中断不再要求从零开始。

## 当前边界

- 语义精度依赖拍摄覆盖、SAM 掩码和 CLIP 文本匹配；阈值需要按场景调整。
- “删除”是从高斯 PLY 中过滤点，不会生成被遮挡背景；删除大物体后可能留下空洞。
- 点击拾取依据已注册相机上的高斯中心投影，是轻量近似，不是逐像素 ID-buffer。
- 极少于 6–8 张图时，COLMAP 本身可能无法估计可靠相机；深度先验可以改善几何，但不能保证恢复未观测区域。

---

# Upstream: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
Bernhard Kerbl*, Georgios Kopanas*, Thomas Leimkühler, George Drettakis (* indicates equal contribution)<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | [Video](https://youtu.be/T_kXY43VZnk) | [Other GRAPHDECO Publications](http://www-sop.inria.fr/reves/publis/gdindex.php) | [FUNGRAPH project page](https://fungraph.inria.fr) |<br>
| [T&T+DB COLMAP (650MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) | [Pre-trained Models (14 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) | [Viewers for Windows (60MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) | [Evaluation Images (7 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip) |<br>
![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models. 

<a href="https://www.inria.fr/"><img height="100" src="assets/logo_inria.png"> </a>
<a href="https://univ-cotedazur.eu/"><img height="100" src="assets/logo_uca.png"> </a>
<a href="https://www.mpi-inf.mpg.de"><img height="100" src="assets/logo_mpi.png"> </a> 
<a href="https://team.inria.fr/graphdeco/"> <img style="width:100%;" src="assets/logo_graphdeco.png"></a>

Abstract: *Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (≥ 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>



## Funding and Acknowledgments

This research was funded by the ERC Advanced grant FUNGRAPH No 788065. The authors are grateful to Adobe for generous donations, the OPAL infrastructure from Université Côte d’Azur and for the HPC resources from GENCI–IDRIS (Grant 2022-AD011013409). The authors thank the anonymous reviewers for their valuable feedback, P. Hedman and A. Tewari for proofreading earlier drafts also T. Müller, A. Yu and S. Fridovich-Keil for helping with the comparisons.

## NEW FEATURES !

We have limited resources for maintaining and updating the code. However, we have added a few new features since the original release that are inspired by some of the excellent work many other researchers have been doing on 3DGS. We will be adding other features within the ability of our resources.

**Update of October 2024**: We integrated [training speed acceleration](#training-speed-acceleration) and made it compatible with [depth regularization](#depth-regularization), [anti-aliasing](#anti-aliasing) and [exposure compensation](#exposure-compensation). We have enhanced the SIBR real time viewer by correcting bugs and adding features in the [Top View](#sibr-top-view) that allows visualization of input and user cameras.

**Update of Spring 2024**:
Orange Labs has kindly added [OpenXR support](#openxr-support) for VR viewing. 

## Step-by-step Tutorial

Jonathan Stephens made a fantastic step-by-step tutorial for setting up Gaussian Splatting on your machine, along with instructions for creating usable datasets from videos. If the instructions below are too dry for you, go ahead and check it out [here](https://www.youtube.com/watch?v=UXtuigy_wYc).

## Colab

User [camenduru](https://github.com/camenduru) was kind enough to provide a Colab template that uses this repo's source (status: August 2023!) for quick and easy access to the method. Please check it out [here](https://github.com/camenduru/gaussian-splatting-colab).

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## Overview

The codebase has 4 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A network viewer that allows to connect to and visualize the optimization process
- An OpenGL-based real-time viewer to render trained models in real-time.
- A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10 and Ubuntu Linux 22.04. Instructions for setting up and running each of them are found in the sections below.




## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gaussian_splatting
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. For modifications, see below.

Tip: Downloading packages and creating a new environment with Conda can require a significant amount of disk space. By default, Conda will use the main system hard drive. You can avoid this by specifying a different package download location and an environment on a different drive:

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

#### Modifications

If you can afford the disk space, we recommend using our environment files for setting up a training environment identical to ours. If you want to make modifications, please note that major version changes might affect the results of our method. However, our (limited) experiments suggest that the codebase works just fine inside a more up-to-date environment (Python 3.8, PyTorch 2.0.0, CUDA 12). Make sure to create an environment where PyTorch and its CUDA runtime version match and the installed CUDA SDK has no major version difference with PyTorch's CUDA version.

#### Known Issues

Some users experience problems building the submodules on Windows (```cl.exe: File not found``` or similar). Please consider the workaround for this problem from the FAQ.

### Running

To run the optimizer, simply use

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

Note that similar to MipNeRF360, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting ```-r 1```.

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can find our SfM data sets for Tanks&Temples and Deep Blending [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). If you do not provide an output model directory (```-m```), trained models are written to folders with randomized unique names inside the ```output``` directory. At this point, the trained models may be viewed with the real-time viewer (see further below).

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

If you want to evaluate our [pre-trained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), you will have to download the corresponding source data sets and indicate their location to ```render.py``` with an additional ```--source_path/-s``` flag. Note: The pre-trained models were created with the release codebase. This code base has been cleaned up and includes bugfixes, hence the metrics you get from evaluating them will differ from those in the paper.
```shell
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

We further provide the ```full_eval.py``` script. This script specifies the routine used in our evaluation and demonstrates the use of some additional parameters, e.g., ```--images (-i)``` to define alternative image directories within COLMAP data sets. If you have downloaded and extracted all the training data, you can run it like this:
```shell
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```
In the current version, this process takes about 7h on our reference machine containing an A6000. If you want to do the full evaluation on our pre-trained models, you can specify their download location and skip training. 
```shell
python full_eval.py -o <directory with pretrained models> --skip_training -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```

If you want to compute the metrics on our paper's [evaluation images](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip), you can also skip rendering. In this case it is not necessary to provide the source datasets. You can compute metrics for multiple image sets at a time. 
```shell
python full_eval.py -m <directory with evaluation images>/garden ... --skip_training --skip_rendering
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for full_eval.py</span></summary>
  
  #### --skip_training
  Flag to skip training stage.
  #### --skip_rendering
  Flag to skip rendering stage.
  #### --skip_metrics
  Flag to skip metrics calculation stage.
  #### --output_path
  Directory to put renderings and results in, ```./eval``` by default, set to pre-trained model location if evaluating them.
  #### --mipnerf360 / -m360
  Path to MipNeRF360 source datasets, required if training or rendering.
  #### --tanksandtemples / -tat
  Path to Tanks&Temples source datasets, required if training or rendering.
  #### --deepblending / -db
  Path to Deep Blending source datasets, required if training or rendering.
</details>
<br>

## Interactive Viewers
We provide two interactive viewers for our method: remote and real-time. Our viewing solutions are based on the [SIBR](https://sibr.gitlabpages.inria.fr/) framework, developed by the GRAPHDECO group for several novel-view synthesis projects.

### Hardware Requirements
- OpenGL 4.5-ready GPU and drivers (or latest MESA software)
- 4 GB VRAM recommended
- CUDA-ready GPU with Compute Capability 7.0+ (only for Real-Time Viewer)

### Software Requirements
- Visual Studio or g++, **not Clang** (we used Visual Studio 2019 for Windows)
- CUDA SDK 11, install *after* Visual Studio (we used 11.8)
- CMake (recent version, we used 3.24)
- 7zip (only on Windows)

### Pre-built Windows Binaries
We provide pre-built binaries for Windows [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip). We recommend using them on Windows for an efficient setup, since the building of SIBR involves several external dependencies that must be downloaded and compiled on-the-fly.

### Installation from Source
If you cloned with submodules (e.g., using ```--recursive```), the source code for the viewers is found in ```SIBR_viewers```. The network viewer runs within the SIBR framework for Image-based Rendering applications.

#### Windows
CMake should take care of your dependencies.
```shell
cd SIBR_viewers
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo
```
You may specify a different configuration, e.g. ```Debug``` if you need more control during development.

#### Ubuntu 22.04
You will need to install a few dependencies before running the project setup.
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
``` 

#### Ubuntu 20.04
Backwards compatibility with Focal Fossa is not fully tested, but building SIBR with CMake should still work after invoking
```shell
git checkout fossa_compatibility
```

### Navigation in SIBR Viewers
The SIBR interface provides several methods of navigating the scene. By default, you will be started with an FPS navigator, which you can control with ```W, A, S, D, Q, E``` for camera translation and ```I, K, J, L, U, O``` for rotation. Alternatively, you may want to use a Trackball-style navigator (select from the floating menu). You can also snap to a camera from the data set with the ```Snap to``` button or find the closest camera with ```Snap to closest```. The floating menues also allow you to change the navigation speed. You can use the ```Scaling Modifier``` to control the size of the displayed Gaussians, or show the initial point cloud.

### Running the Network Viewer



https://github.com/graphdeco-inria/gaussian-splatting/assets/40643808/90a2e4d3-cf2e-4633-b35f-bfe284e28ff7



After extracting or installing the viewers, you may run the compiled ```SIBR_remoteGaussian_app[_config]``` app in ```<SIBR install dir>/bin```, e.g.: 
```shell
./<SIBR install dir>/bin/SIBR_remoteGaussian_app
```
The network viewer allows you to connect to a running training process on the same or a different machine. If you are training on the same machine and OS, no command line parameters should be required: the optimizer communicates the location of the training data to the network viewer. By default, optimizer and network viewer will try to establish a connection on **localhost** on port **6009**. You can change this behavior by providing matching ```--ip``` and ```--port``` parameters to both the optimizer and the network viewer. If for some reason the path used by the optimizer to find the training data is not reachable by the network viewer (e.g., due to them running on different (virtual) machines), you may specify an override location to the viewer by using ```-s <source path>```. 

<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Network Viewer</span></summary>

  #### --path / -s
  Argument to override model's path to source dataset.
  #### --ip
  IP to use for connection to a running training script.
  #### --port
  Port to use for connection to a running training script. 
  #### --rendering-size 
  Takes two space separated numbers to define the resolution at which network rendering occurs, ```1200``` width by default.
  Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.
  #### --load_images
  Flag to load source dataset images to be displayed in the top view for each camera.
</details>
<br>

### Running the Real-Time Viewer




https://github.com/graphdeco-inria/gaussian-splatting/assets/40643808/0940547f-1d82-4c2f-a616-44eabbf0f816




After extracting or installing the viewers, you may run the compiled ```SIBR_gaussianViewer_app[_config]``` app in ```<SIBR install dir>/bin```, e.g.: 
```shell
./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model>
```

It should suffice to provide the ```-m``` parameter pointing to a trained model directory. Alternatively, you can specify an override location for training input data using ```-s```. To use a specific resolution other than the auto-chosen one, specify ```--rendering-size <width> <height>```. Combine it with ```--force-aspect-ratio``` if you want the exact resolution and don't mind image distortion. 

**To unlock the full frame rate, please disable V-Sync on your machine and also in the application (Menu &rarr; Display). In a multi-GPU system (e.g., laptop) your OpenGL/Display GPU should be the same as your CUDA GPU (e.g., by setting the application's GPU preference on Windows, see below) for maximum performance.**

![Teaser image](assets/select.png)

In addition to the initial point cloud and the splats, you also have the option to visualize the Gaussians by rendering them as ellipsoids from the floating menu.
SIBR has many other functionalities, please see the [documentation](https://sibr.gitlabpages.inria.fr/) for more details on the viewer, navigation options etc. There is also a Top View (available from the menu) that shows the placement of the input cameras and the original SfM point cloud; please note that Top View slows rendering when enabled. The real-time viewer also uses slightly more aggressive, fast culling, which can be toggled in the floating menu. If you ever encounter an issue that can be solved by turning fast culling off, please let us know.

<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Real-Time Viewer</span></summary>

  #### --model-path / -m
  Path to trained model.
  #### --iteration
  Specifies which of state to load if multiple are available. Defaults to latest available iteration.
  #### --path / -s
  Argument to override model's path to source dataset.
  #### --rendering-size 
  Takes two space separated numbers to define the resolution at which real-time rendering occurs, ```1200``` width by default. Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.
  #### --load_images
  Flag to load source dataset images to be displayed in the top view for each camera.
  #### --device
  Index of CUDA device to use for rasterization if multiple are available, ```0``` by default.
  #### --no_interop
  Disables CUDA/GL interop forcibly. Use on systems that may not behave according to spec (e.g., WSL2 with MESA GL 4.5 software rendering).
</details>
<br>

## Processing your own Scenes

Our COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. We provide a converter script ```convert.py```, to extract undistorted images and SfM information from input images. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders. To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. Put the images you want to use in a directory ```<location>/input```.
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
 If you have COLMAP and ImageMagick on your system path, you can simply run 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```
Alternatively, you can use the optional parameters ```--colmap_executable``` and ```--magick_executable``` to point to the respective paths. Please note that on Windows, the executable should point to the COLMAP ```.bat``` file that takes care of setting the execution environment. Once done, ```<location>``` will contain the expected COLMAP data set structure with undistorted, resized input images, in addition to your original images and some temporary (distorted) data in the directory ```distorted```.

If you have your own COLMAP dataset without undistortion (e.g., using ```OPENCV``` camera), you can try to just run the last part of the script: Put the images in ```input``` and the COLMAP info in a subdirectory ```distorted```:
```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
Then run 
```shell
python convert.py -s <location> --skip_matching [--resize] #If not resizing, ImageMagick is not needed
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for convert.py</span></summary>

  #### --no_gpu
  Flag to avoid using GPU in COLMAP.
  #### --skip_matching
  Flag to indicate that COLMAP info is available for images.
  #### --source_path / -s
  Location of the inputs.
  #### --camera 
  Which camera model to use for the early matching steps, ```OPENCV``` by default.
  #### --resize
  Flag for creating resized versions of input images.
  #### --colmap_executable
  Path to the COLMAP executable (```.bat``` on Windows).
  #### --magick_executable
  Path to the ImageMagick executable.
</details>
<br>

### Training speed acceleration

We integrated the drop-in replacements from [Taming-3dgs](https://humansensinglab.github.io/taming-3dgs/)<sup>1</sup> with [fused ssim](https://github.com/rahul-goel/fused-ssim/tree/main) into the original codebase to speed up training times. Once installed, the accelerated rasterizer delivers a **$\times$ 1.6 training time speedup** using `--optimizer_type default` and a **$\times$ 2.7 training time speedup** using `--optimizer_type sparse_adam`.

To get faster training times you must first install the accelerated rasterizer to your environment:

```bash
pip uninstall diff-gaussian-rasterization -y
cd submodules/diff-gaussian-rasterization
rm -r build
git checkout 3dgs_accel
pip install .
```

Then you can add the following parameter to use the sparse adam optimizer when running `train.py`:

```bash
--optimizer_type sparse_adam
```

*Note that this custom rasterizer has a different behaviour than the original version, for more details on training times please see [stats for training times](results.md/#training-times-comparisons)*.

*1. Mallick and Goel, et al. ‘Taming 3DGS: High-Quality Radiance Fields with Limited Resources’. SIGGRAPH Asia 2024 Conference Papers, 2024, https://doi.org/10.1145/3680528.3687694, [github](https://github.com/humansensinglab/taming-3dgs)*


### Depth regularization

To have better reconstructed scenes we use depth maps as priors during optimization with each input images. It works best on untextured parts ex: roads and can remove floaters. Several papers have used similar ideas to improve various aspects of 3DGS; (e.g. [DepthRegularizedGS](https://robot0321.github.io/DepthRegGS/index.html), [SparseGS](https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/), [DNGaussian](https://fictionarry.github.io/DNGaussian/)). The depth regularization we integrated is that used in our [Hierarchical 3DGS](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) paper, but applied to the original 3DGS; for some scenes (e.g., the DeepBlending scenes) it improves quality significantly; for others it either makes a small difference or can even be worse. For example results showing the potential benefit and statistics on quality please see here: [Stats for depth regularization](results.md).

When training on a synthetic dataset, depth maps can be produced and they do not require further processing to be used in our method.

For real world datasets depth maps should be generated for each input images, to generate them please do the following:
1. Clone [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#usage):
    ```
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    ```
2. Download weights from [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it under `Depth-Anything-V2/checkpoints/`
3. Generate depth maps:
   ```
   python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path <path to input images> --outdir <output path>
   ```
5. Generate a `depth_params.json` file using:
    ```
    python utils/make_depth_scale.py --base_dir <path to colmap> --depths_dir <path to generated depths>
    ```

A new parameter should be set when training if you want to use depth regularization `-d <path to depth maps>`.

### Exposure compensation
To compensate for exposure changes in the different input images we optimize an affine transformation for each image just as in [Hierarchical 3dgs](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/).  

This can greatly improve reconstruction results for "in the wild" captures, e.g., with a smartphone when the exposure setting of the camera is not fixed. For example results showing the potential benefit and statistics on quality please see here: [Stats for exposure compensation](results.md).

Add the following parameters to enable it:
```
--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp
```
Again, other excellent papers have used similar ideas e.g. [NeRF-W](https://nerf-w.github.io/), [URF](https://urban-radiance-fields.github.io/).

### Anti-aliasing
We added the EWA Filter from [Mip Splatting](https://niujinshuchong.github.io/mip-splatting/) in our codebase to remove aliasing. It is disabled by default but you can enable it by adding `--antialiasing` when training on a scene using `train.py` or rendering using `render.py`. Antialiasing can be toggled in the SIBR viewer, it is disabled by default but you should enable it when viewing a scene trained using `--antialiasing`.
![aa](/assets/aa_onoff.gif)
*this scene was trained using `--antialiasing`*.

### SIBR: Top view
> `Views > Top view`

The `Top view` renders the SfM point cloud in another view with the corresponding input cameras and the `Point view` user camera. This allows visualization of how far the viewer is from the input cameras for example.

It is a 3D view so the user can navigate through it just as in the `Point view` (modes available: FPS, trackball, orbit).
<!-- _gif showing the top view, showing it is realtime_ -->
<!-- ![topViewOpen_1.gif](../assets/topViewOpen_1_1709560483017_0.gif) -->
![top view open](assets/top_view_open.gif)

Options are available to customize this view, meshes can be disabled/enabled and their scales can be modified. 
<!-- _gif showing different options_ -->
<!-- ![topViewOptions.gif](../assets/topViewOptions_1709560615266_0.gif) -->
![top view options](assets/top_view_options.gif)
A useful additional functionality is to move to the position of an input image, and progressively fade out to the SfM point view in that position (e.g., to verify camera alignment). Views from input cameras can be displayed in the `Top view` (*note that `--images-path` must be set in the command line*). One can snap the `Top view` camera to the closest input camera from the user camera in the `Point view` by clicking `Top view settings > Cameras > Snap to closest`. 
<!-- _gif showing for a snapped camera the ground truth image with alpha_ -->
<!-- ![topViewImageAlpha.gif](../assets/topViewImageAlpha_1709560852268_0.gif) -->
![top view image alpha](assets/top_view_image_alpha.gif)

### OpenXR support

OpenXR is supported in the branch `gaussian_code_release_openxr` 
Within that branch, you can find documentation for VR support [here](https://gitlab.inria.fr/sibr/sibr_core/-/tree/gaussian_code_release_openxr?ref_type=heads).


## FAQ
- *Where do I get data sets, e.g., those referenced in ```full_eval.py```?* The MipNeRF360 data set is provided by the authors of the original paper on the project site. Note that two of the data sets cannot be openly shared and require you to consult the authors directly. For Tanks&Temples and Deep Blending, please use the download links provided at the top of the page. Alternatively, you may access the cloned data (status: August 2023!) from [HuggingFace](https://huggingface.co/camenduru/gaussian-splatting)


- *How can I use this for a much larger dataset, like a city district?* The current method was not designed for these, but given enough memory, it should work out. However, the approach can struggle in multi-scale detail scenes (extreme close-ups, mixed with far-away shots). This is usually the case in, e.g., driving data sets (cars close up, buildings far away). For such scenes, you can lower the ```--position_lr_init```, ```--position_lr_final``` and ```--scaling_lr``` (x0.3, x0.1, ...). The more extensive the scene, the lower these values should be. Below, we use default learning rates (left) and ```--position_lr_init 0.000016 --scaling_lr 0.001"``` (right).

| ![Default learning rate result](assets/worse.png "title-1") <!-- --> | <!-- --> ![Reduced learning rate result](assets/better.png "title-2") |
| --- | --- |

- *I'm on Windows and I can't manage to build the submodules, what do I do?* Consider following the steps in the excellent video tutorial [here](https://www.youtube.com/watch?v=UXtuigy_wYc), hopefully they should help. The order in which the steps are done is important! Alternatively, consider using the linked Colab template.

- *It still doesn't work. It says something about ```cl.exe```. What do I do?* User Henry Pearce found a workaround. You can you try adding the visual studio path to your environment variables (your version number might differ);
```C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64```
Then make sure you start a new conda prompt and cd to your repo location and try this;
```
conda activate gaussian_splatting
cd <dir_to_repo>/gaussian-splatting
pip install submodules\diff-gaussian-rasterization
pip install submodules\simple-knn
```

- *I'm on macOS/Puppy Linux/Greenhat and I can't manage to build, what do I do?* Sorry, we can't provide support for platforms outside of the ones we list in this README. Consider using the linked Colab template.

- *I don't have 24 GB of VRAM for training, what do I do?* The VRAM consumption is determined by the number of points that are being optimized, which increases over time. If you only want to train to 7k iterations, you will need significantly less. To do the full training routine and avoid running out of memory, you can increase the ```--densify_grad_threshold```, ```--densification_interval``` or reduce the value of ```--densify_until_iter```. Note however that this will affect the quality of the result. Also try setting ```--test_iterations``` to ```-1``` to avoid memory spikes during testing. If ```--densify_grad_threshold``` is very high, no densification should occur and training should complete if the scene itself loads successfully.

- *24 GB of VRAM for reference quality training is still a lot! Can't we do it with less?* Yes, most likely. By our calculations it should be possible with **way** less memory (~8GB). If we can find the time we will try to achieve this. If some PyTorch veteran out there wants to tackle this, we look forward to your pull request!


- *How can I use the differentiable Gaussian rasterizer for my own project?* Easy, it is included in this repo as a submodule ```diff-gaussian-rasterization```. Feel free to check out and install the package. It's not really documented, but using it from the Python side is very straightforward (cf. ```gaussian_renderer/__init__.py```).

- *Wait, but ```<insert feature>``` isn't optimized and could be much better?* There are several parts we didn't even have time to think about improving (yet). The performance you get with this prototype is probably a rather slow baseline for what is physically possible.

- *Something is broken, how did this happen?* We tried hard to provide a solid and comprehensible basis to make use of the paper's method. We have refactored the code quite a bit, but we have limited capacity to test all possible usage scenarios. Thus, if part of the website, the code or the performance is lacking, please create an issue. If we find the time, we will do our best to address it.
