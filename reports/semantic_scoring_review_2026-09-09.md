# Ramen 语义评分与论文实现核对

本报告是代码与协议审查，不是新模型精度报告。已有 v3 同口径 GPU 结果为
mIoU 19.2962%、Boundary-IoU 8.0318%，其中筷子为 0、黄色碗约 58.10%。
这些结果确实说明语义仍弱，但不能只凭数值断定全部错误都来自训练或评分。
本轮新增可选评分和评估接口；新接口尚待真实 GPU 对照验证。

## 核实到的实现差异

| 参考实现 | 官方做法 | 本仓库旧版与本轮处理 |
|---|---|---|
| LangSplat | 特征还原到 CLIP 空间后，与目标文本和通用负文本比较，取最难负文本的二分类相关度 | 旧版直接在中心化 PCA 空间算 cosine；新增 `clip_cosine` 和 `clip_relevancy`，默认保留 `legacy_pca_cosine` |
| Gaussian Grouping 的 LERF-Mask 评估 | 预测 mask 对齐到原始 GT 分辨率；边界比例 0.02，补零边界；先类别平均再宏平均 | 旧版把 GT 缩到渲染尺寸、比例 0.008、不补边界、平均所有视角/类别行；新增显式 `gg_native` |
| SAGA | 独立亲和特征结合尺度条件进行区域对应训练，用于交互分割 | 旧版把门控直接用于语言特征；v5 将亲和与语言分支分开，推理分别处理，不把分组维度冒充物理尺度复现 |
| LaGa | 先分解 3D 对象；对象级多视角描述子自适应聚类，按权重聚合相关度 | 全局原型或单个 PCA 语言向量并不等价；本轮只修评分和预留独立亲和接口，尚未完整复现对象级描述子数据库 |

上述依据分别是官方
[LangSplat 评分代码](https://github.com/minghanqin/LangSplat/blob/main/eval/openclip_encoder.py)、
[Gaussian Grouping 评估脚本](https://github.com/lkeab/gaussian-grouping/blob/main/script/eval_lerf_mask.py)、
[SAGA 训练代码](https://github.com/Jumpat/SegAnyGAussians/blob/v2/train_contrastive_feature.py) 与
[LaGa 官方仓库](https://github.com/SJTU-DeepVisionLab/LaGa)。

LangSplat 的完整评估还包含平滑、逐图分数变换与按最大相关度选择层级；其官方
标注读取的是多边形 JSON，不能把它的分数直接当成本任务 LERF-Mask PNG 的
同协议基线。本轮不复制逐测试图 min/max 后调阈值，也不利用测试 GT 选层。
[官方评估实现](https://github.com/minghanqin/LangSplat/blob/main/eval/evaluate_iou_loc.py)

Gaussian Grouping 的文本到 mask 路径先借助 Grounded-SAM 选择实例，再由
3D 实例概率生成 mask，实例概率阈值不是 CLIP cosine 阈值。因此本轮只对齐
其最终二值 mask 的计分规则，不能宣称复现了它的训练或文本查询能力。
[官方 mask 渲染实现](https://github.com/lkeab/gaussian-grouping/blob/main/render_lerf_mask.py)

LaGa GUI 将描述子相关度乘聚类权重，在每对象内取最大值，再聚合层级；还提供
过滤后处理。这支持“对象级多视角描述子优于把所有区域硬压成一个原型”的
设计方向，但是否改善本场景需独立消融，不能据引用直接声称已经提升。
[官方相关度聚合实现](https://github.com/SJTU-DeepVisionLab/LaGa/blob/main/laga_gui.py)

## 评分问题的具体证据与修复

设 PCA 表示为 `z=(x−μ)Cᵀ`。即使没有降维，只做中心化也会改变 cosine。
数值测试用 `μ=(0.8,0.2)`、`z=(-0.6,0.6)`、文本 `t=(1,0)`：旧 PCA
cosine 为 −1，恢复到 `x=(0.2,0.8)` 后 CLIP cosine 约为 0.2425。
因此“旧 cosine 大于 0.25”不是“原 CLIP cosine 大于 0.25”。

新增 `clip_space_cosines` 计算 `zC+μ` 与文本的真实 cosine，利用小型 Gram
矩阵避免为百万高斯展开完整 CLIP 维度。它只恢复 PCA 的可重构部分，不可能
恢复被丢弃的信息。新增 `clip_relevancy` 使用固定负词
`object / things / stuff / texture`，分数为
`sigmoid(10 × (目标 cosine − 最大负词 cosine))`，默认阈值 0.5。
这是固定初始规则，不是已经在 Ramen 验证集选出的最优阈值，也不是校准概率。

新像素评分先合成语言特征、除以前景 alpha、还原 CLIP 空间再评分；旧版是
先对每高斯 cosine 截断至非负再合成，二者因非线性操作不可交换。新路径排除
alpha 小于 `1e-4` 的未覆盖像素，防止低覆盖区域仅由 PCA 均值生成有效语义。
旧协议仍保持原来的逐高斯评分、阈值 0.25、粒度 1 和边界比例 0.008。

潜在真实训练问题仍包括 SAM 对细长物体漏检、CLIP 裁剪背景污染、跨视角物体
对应错误、层级边界混合、旧 Y 投影错误与监督分辨率不足。评分修复不是这些
问题的替代品，应查看已保存的 GT/预测 mask 叠加图逐类定位。

## 可复现测试与结果隔离

先用同一 v3 权重跑旧评分旧协议作回归，接着用新评分在旧 mask 协议下消融，
最后在 `gg_native` 下对本模型和公开预测 mask 统一重算。每组使用独立输出
目录，不覆盖已核验 v3 结果，不用最终测试标签选择阈值或层级。

```bash
# 旧协议回归；model 和 test_mask 填实际目录
python -m scripts.evaluate_lerf_mask --model MODEL --test_mask MASKS \
  --iteration 15000 --score_mode legacy_pca_cosine --mask_protocol legacy \
  --threshold 0.25 --granularity 1 --boundary_ratio 0.008 --output EVAL_LEGACY

# 新固定评分 + 官方 GG mask 计分几何，不等于 LangSplat/GG 模型复现
python -m scripts.evaluate_lerf_mask --model MODEL --test_mask MASKS \
  --iteration 15000 --score_mode clip_relevancy --mask_protocol gg_native \
  --threshold 0.5 --granularity 1 --boundary_ratio 0.02 --output EVAL_CLIP_GG
```

如需只检查评分因素，第二组改用 `--mask_protocol legacy --boundary_ratio 0.008`
并另存目录。报告生成器现在禁止在评分空间、负词、alpha、聚合或 mask 协议
不同的结果之间自动声称提升。双方同协议也仍需检查数据、训练预算和权重来源。

## v5 交互层级最小接口

`semantic_query.py` 新增点选查询，不加载 CLIP：

```bash
python semantic_query.py --model MODEL_V5 --iteration 15000 \
  --affinity_point_index 123 --granularity 2 --threshold 0.7 \
  --output affinity_selection.npz --json affinity_selection.json
```

点编号必须来自该次导出的高斯顺序，不能拿旧模型的编号查询新模型。v5 工件的
`features[N,32]` 仍是语言表示；`affinity_features[N,16]` 是独立亲和表示。
粗/中/细层分别取 8/12/16 维前缀后整体 L2 归一化，再与点提示比较，不执行
PCA、min/max 解码或语言门控。这提供可测试的推理接口，不是已有 UI 自动
点选连通，也未证明三层已经学出预期粒度。旧模型缺少亲和分支会明确报错。

## 当前验证状态

本地执行评分、评估协议和报告三组定向测试共 22 项：21 通过，1 项因当前
运行环境缺 OpenCV 跳过（原生 GT 边界 fixture）。数值测试覆盖 PCA 还原、
不修改文本输入、最难负词、旧评分不变、独立亲和前缀及协议比较防护。
该数字与先前 Colab 已核验的 79 项测试分开记录；新评分尚无 GPU 质量结果。
后续应在具有 OpenCV/CUDA 的 Colab 执行完整测试和上述同权重消融，保留
真实输出后再更新本文。
