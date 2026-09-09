# Ramen v5：正式实验完成核验（2026-09-09）

冻结训练代码：`8cae9836de3aa185f807f22031a188e5d6db54d8`。以下正文来自已校验的 Drive 最终报告，不是旧 v3 或 300 步 smoke。

## 后续状态核验

2026-09-09 重新挂载云盘后，已核对 `ramen_v5_live_snapshot/snapshot_manifest.json`：最后成功保存于 `2026-09-09T08:45:11.584868+00:00`，436 个登记文件，pending 为 null，teacher.state 为 complete。最新 `output/monitored_state.json` 与清单 SHA 一致，明确记录 `status=complete`、`stage=complete`、`computation_complete=true`；benchmark、descriptor_bank、descriptor_evaluation、report 均返回 0。

**训练和评测已完成，无需重训；但 `full_final_archive_completed=false`，不能声称完整权重已经归档或上传。** 本次只发布报告，未重新启动 GPU 训练，也没有删除云盘文件。当前 Colab 后端是新的空训练环境；以下结果来自原后端留下的云盘备份。

主 comparison、三份 metrics、training_times、双方 validation_summary、顺序语义 training_complete、原报告、evidence 与最新 monitor 均已分别核对快照清单 SHA。双方主评估的数据指纹一致，均为 3 个标注测试视角。

## 结论先读

- 主协议下，联合模型 PSNR 29.259249 dB，低于顺序基线 29.764179 dB；mIoU 20.671952% 与基线 20.789650% 接近且略低。主模型没有全面超过基线。
- 训练后描述符库 mIoU 27.413093%、Boundary-IoU 23.902779%，属于独立后处理协议。平均分升高主要由水杯驱动；筷子及其余四类 IoU 低于原联合检索，不代表普遍增强或达到 SOTA。
- 联合实际训练 7283.384353 秒，顺序 RGB＋语义 6737.208092 秒，相差约 546 秒，超出约 146 秒容差；`equal_wall_clock=false`，不是严格等时间比较。
- 下方自动报告里的 `stage=report` / `computation_complete=false` 是报告生成时的历史快照。报告结束后监督器才写最终 complete 状态，不能把旧快照当作当前未完成，也不能混用两个 monitor 版本的 SHA。
- 原始证据仍位于 `MyDrive/semantic_adaptive_3dgs/ramen_v5_live_snapshot/output`。本文件不是所有原始 JSON、预测图或完整模型权重的上传包。

---

# Ramen 联合重建与语义分割实验报告

本报告仅汇总已读取的实验文件。pending 表示证据尚缺，不代表零分、失败或无提升。

## 实验结论与完成状态

双方原始评估包含所列核心指标；具体结论仍受下述协议与审计限制。

单次 PSNR 微小变化不能证明稳定提升；高斯数量下降也不能单独证明算法更有效。

## 标注视角评估结果（协议核验见下文）

以下数值来自 eval_*/metrics.json，或明确标记的 comparison.json 摘要；只覆盖该评估器记录的标注视角。

| 指标 | 本次联合模型 | 本次顺序基线 | 历史联合模型 |
|---|---:|---:|---:|
| PSNR (dB) ↑ | 29.259249 | 29.764179 | pending（未取得证据） |
| SSIM ↑ | 0.924395 | 0.930271 | pending（未取得证据） |
| mIoU ↑ | 0.206720 | 0.207897 | pending（未取得证据） |
| Boundary-IoU ↑ | 0.178020 | 0.174663 | pending（未取得证据） |
| 高斯数量 | 1,972,127 | 1,393,877 | pending（未取得证据） |

| 运行 | 指标来源类型 | 标注重建视角数 | 迭代标签 |
|---|---|---:|---:|
| 本次联合 | annotated_evaluator | 3 | 15,000 |
| 本次顺序 | annotated_evaluator | 3 | 15,000 |
| 历史联合 | missing | pending（未取得证据） | pending（未取得证据） |

### 对比资格与改变量

- 本次联合 − 本次顺序：recorded_protocol_matches。
  已记录口径下的原始差值：PSNR (dB) ↑ -0.504930；SSIM ↑ -0.005876；mIoU ↑ -0.001177；Boundary-IoU ↑ +0.003358；高斯数量 +578,250。这不是统计显著性结论。
- 本次联合 − 历史联合：protocol_mismatch。
  缺少：双方原始 annotated evaluator metrics.json, negative_prompts, relevancy_temperature, reconstruction_views, mask_views_and_labels, threshold, granularity。
  协议不同：score_mode, mask_metric_protocol, score_compositing, score_space, alpha_min, boundary_pad_edges, iou_aggregation，不计算提升。
  Boundary-IoU 的 boundary_ratio 未完整记录，禁止计算该项改变量。
  评估器版本未完整记录；匹配仅指已记录的视角和参数，不能证明实现完全相同。
  数据/标注文件指纹未完整记录，不能认证不同运行的数据内容完全相同。

## 训练日志与验证集（独立口径）

训练日志 all-test PSNR 与上述 annotated evaluator PSNR 不混用，不跨口径相减。

| 运行 | 日志指标 | 数值 | 迭代 | 范围 | 证据来源 |
|---|---|---:|---:|---|---|
| — | pending（未取得证据） | — | — | — | — |

| 运行 | 记录的最佳验证 PSNR | 最佳迭代 | 实际训练迭代 | 导出迭代别名 |
|---|---:|---:|---:|---:|
| 本次联合 | 24.146916 | 15,000 | 15,000 | 15,000 |
| 本次顺序 | 23.657487 | 8,000 | 12,000 | 15,000 |
| 历史联合 | pending（未取得证据） | pending（未取得证据） | pending（未取得证据） | pending（未取得证据） |

best_psnr / best_iteration 是训练程序记录的选择结果，可能受 min_delta 约束，不一定等于 history 数学最大值。最佳 checkpoint、末轮 checkpoint 和导出别名必须区分；以上表格不把不同 checkpoint 的分数合并。

## 三级重要性与逐类别结果

| 模型 | 级别 | 该级标签 | 区域 PSNR | 类别平均 IoU | 高斯数量 |
|---|---|---|---:|---:|---:|
| 本次联合模型 | 重要物品 | egg, pork belly, wavy noodles in bowl | 31.483243 | 0.148019 | 266,240 |
| 本次联合模型 | 普通物品 | yellow bowl, chopsticks, glass of water | 29.669530 | 0.265420 | 1,311,989 |
| 本次联合模型 | 背景 | table, wall | pending（未取得证据） | pending（未取得证据） | 393,898 |
| 本次顺序基线 | 重要物品 | egg, pork belly, wavy noodles in bowl | 31.803711 | 0.172581 | pending（未取得证据） |
| 本次顺序基线 | 普通物品 | yellow bowl, chopsticks, glass of water | 29.430978 | 0.243212 | pending（未取得证据） |
| 本次顺序基线 | 背景 | table, wall | pending（未取得证据） | pending（未取得证据） | pending（未取得证据） |

背景是剩余区域，不自动等同于一个有完整人工标注的语义类别；不以 1 − 前景 IoU 推算背景 IoU。

| 类别 | 联合 IoU | 顺序 IoU | 历史联合 IoU | 联合 Boundary-IoU | 顺序 Boundary-IoU |
|---|---:|---:|---:|---:|---:|
| chopsticks | 0.057272 | 0.062919 | pending（未取得证据） | 0.057301 | 0.063432 |
| egg | 0.102846 | 0.093740 | pending（未取得证据） | 0.076902 | 0.070967 |
| glass of water | 0.051651 | 0.016099 | pending（未取得证据） | 0.056528 | 0.018075 |
| pork belly | 0.111144 | 0.131991 | pending（未取得证据） | 0.114585 | 0.128883 |
| wavy noodles in bowl | 0.230066 | 0.292011 | pending（未取得证据） | 0.165425 | 0.200707 |
| yellow bowl | 0.687337 | 0.650619 | pending（未取得证据） | 0.597380 | 0.565913 |

## 训练用时与公平性

不能认证等时间对比。实测用时超出容差。

联合训练：7283.384353 秒；顺序 RGB＋语义：6737.208092 秒；差值：-546.176261 秒。

断点续训前的时长缺失、阶段未完成或仅传入 --equal_time 均不能认证等时间。预处理、评估、GPU型号与重复试验耗时需另外记录，阶段用时不自动等于端到端成本。

## 独立附表：联合模型＋训练后描述符库

这是额外的训练后构库与检索评估协议，不替换主表联合模型，不纳入主等时间比较，亦不自动计算相对基线的提升。构库、检索和额外评测成本需单独计入端到端预算。

以下仅是已记录分数；不表示新模型已优于基线，也不能由短训练连通测试证明质量。

| 指标（标注视角） | 描述符库独立评估 |
|---|---:|
| PSNR (dB) ↑ | 29.259249 |
| SSIM ↑ | 0.924395 |
| mIoU ↑ | 0.274131 |
| Boundary-IoU ↑ | 0.239028 |
| 高斯数量 | 1,972,127 |

| 记录项目 | 证据 |
|---|---|
| 独立指标文件 | /content/ramen_semantic_v5/outputs_full/eval_joint_descriptor_bank/metrics.json |
| 评分 / mask 协议 | {"mask_metric_protocol": "gg_native", "score_compositing": "render_signed_affinity_then_descriptor_bank_score", "score_mode": "descriptor_bank", "score_space": "raw_clip_region_descriptors_and_independent_affinity"} |
| 最终二值 mask 阈值 | 0.25 |
| 固定粒度 | 1 |
| Boundary 比例 | 0.02 |
| 候选检索参数（含独立文本门限） | {"affinity_threshold": 0.7, "aggregation": "reliability_weighted_max_with_distinct_view_support", "cross_view_threshold": 0.8, "level": 1, "max_candidates": 64, "min_views": 2, "score_mode": "descriptor_bank", "temperature": 10.0, "text_threshold": 0.5} |
| 标注重建视角 | [["0", "test_0.jpg"], ["1", "test_1.jpg"], ["2", "test_2.jpg"]] |
| 标注 mask 视角与类别 | [["0", "test_0.jpg", "chopsticks"], ["0", "test_0.jpg", "egg"], ["0", "test_0.jpg", "glass of water"], ["0", "test_0.jpg", "pork belly"], ["0", "test_0.jpg", "wavy noodles in bowl"], ["0", "test_0.jpg", "yellow bowl"], ["1", "test_1.jpg", "chopsticks"], ["1", "test_1.jpg", "egg"], ["1", "test_1.jpg", "glass of water"], ["1", "test_1.jpg", "pork belly"], ["1", "test_1.jpg", "wavy noodles in bowl"], ["1", "test_1.jpg", "yellow bowl"], ["2", "test_2.jpg", "chopsticks"], ["2", "test_2.jpg", "egg"], ["2", "test_2.jpg", "glass of water"], ["2", "test_2.jpg", "pork belly"], ["2", "test_2.jpg", "wavy noodles in bowl"], ["2", "test_2.jpg", "yellow bowl"]] |
| 数据指纹 | 881f73b83eca791a4c1d41021fb762c3ec732e39ae195436bab6269e9b44dbdb |
| 描述符库 SHA-256 | 55037b5e8f0b4c183ba67d6cc57bc7e6b74475e3a5e085ed697507bca28076b9 |
| 构库实际来源视角（记录值，非额外验证） | ["frame_00003.jpg", "frame_00009.jpg", "frame_00014.jpg", "frame_00020.jpg", "frame_00025.jpg", "frame_00031.jpg", "frame_00036.jpg", "frame_00042.jpg", "frame_00047.jpg", "frame_00053.jpg", "frame_00058.jpg", "frame_00064.jpg", "frame_00068.jpg", "frame_00074.jpg", "frame_00079.jpg", "frame_00085.jpg", "frame_00090.jpg", "frame_00096.jpg", "frame_00101.jpg", "frame_00107.jpg", "frame_00112.jpg", "frame_00118.jpg", "frame_00123.jpg", "frame_00129.jpg"] |
| 构库采样参数 | {"iteration": 15000, "max_pixels_per_region": 512, "max_records": 8192, "max_regions_per_view": 96, "max_views": 24, "min_alpha": 0.05, "min_coherence": 0.1, "min_coverage": 0.35, "model": "/content/ramen_semantic_v5/outputs_full/joint", "output": "/content/ramen_semantic_v5/outputs_full/joint/descriptor_bank_15000.npz"} |

最终 mask 阈值与候选 CLIP 文本门限含义不同；不得把不同评分协议、粒度或测试集调参后的数字视为同口径提升。

| 类别 | 描述符库 IoU | 描述符库 Boundary-IoU |
|---|---:|---:|
| chopsticks | 0.010377 | 0.015400 |
| egg | 0.096313 | 0.082103 |
| glass of water | 0.876466 | 0.752322 |
| pork belly | 0.102409 | 0.078130 |
| wavy noodles in bowl | 0.092364 | 0.078583 |
| yellow bowl | 0.466857 | 0.427628 |

all-test RGB（独立范围，不替换标注视角指标）：视角数 4；PSNR 29.606647；SSIM 0.927097。

| 额外成本记录 | 秒数 |
|---|---:|
| 库内构建 elapsed_seconds（构库器记录） | 14.724817 |
| 构库阶段：监控最近成功记录的观测时长 | 30.001423 |
| 库检索评估阶段：监控最近成功记录的观测时长 | 60.003726 |
| 完整端到端额外成本 | pending（未取得证据） |

监控观测时长可能包含轮询和快照开销，不是纯 GPU 时间，也不能认作所有断点续跑尝试的累计时间；构库器耗时与监控构库耗时范围重叠，禁止相加。已有库跳过构建或没有完整记录时，未知成本不是零。主预算只使用 training_times.json 的完整训练阶段计时。

监控状态只是报告生成时的快照，不据此断言之后的阶段已完成或完整权重已归档：{"computation_complete": false, "full_final_archive_completed": false, "stage": "report", "status": "running"}。

## 审计限制与已知问题

- 单场景、单随机种子的结果不能直接代表稀疏视角、单图补全或其他场景能力。
- 伪标签训练损失下降不等同于人工标注 mIoU 提升；边界与跨视角指标需要各自评估。
- 验证集必须与 PCA / 聚类 / 原型构建等学习步骤隔离；若共享拟合，则须标明 transductive，不能作为完全独立验证。

## 可执行改进建议

1. 先修复并验证正确性：对重要性投影、遮挡与像素坐标做可视化叠加；核对 CUDA 实际支持的 SH 阶数，避免无效参数占用显存。
2. 固定相同的人工标注测试视角、图像分辨率、mask 文件、阈值、粒度、boundary_ratio 与评估代码版本，补齐联合和顺序模型的同口径评估。
3. 将预处理拟合限制到训练视角，并冻结验证/测试变换。对 RGB 预热 0 / 1500 / 3000 轮做验证集选择；测试集仅用于最终一次确认。
4. 单独消融边界加权与细长物体致密化：保持可比高斯预算，记录每类 IoU、Boundary-IoU、重要物体区域 PSNR，以及边缘误删/漏检图。
5. 检查跨视角正样本的遮挡和置信度，剔除不可靠对应；分别消融跨视角项与语义项，检验是否改善人工标注而非仅降低伪标签损失。
6. 修复后至少运行多个随机种子，报告均值与离散度；在同一GPU记录完整训练阶段累计时间，再开展等时间对照。

上述为待验证改进方案，不代表已经完成训练或取得提升。

## 云盘清理记录

pending（未取得证据）；未取得清理清单，不能声称已释放云盘空间。

## 产物与原始证据

- markdown: `/content/ramen_semantic_v5/outputs_full/report/ramen_final_report.md`
- evidence_json: `/content/ramen_semantic_v5/outputs_full/report/ramen_evidence.json`
- 输入：`/content/ramen_semantic_v5/outputs_full/comparison.json`；SHA-256：`ce3c5822250cf738a462aea77f23723f100bf564002a99f9253b15b219dceab5`
- 输入：`/content/ramen_semantic_v5/outputs_full/training_times.json`；SHA-256：`2645b1dcf49a3eb0d6962d320de1c1eeaadf5224f3c03ba4eba772ab4327c1b5`
- 输入：`/content/ramen_semantic_v5/outputs_full/eval_joint/metrics.json`；SHA-256：`62cb7b6fb9ed495ffefe9a1c5bdc0878fcdea1f2c3f006c302baf8079ac5a91c`
- 输入：`/content/ramen_semantic_v5/outputs_full/joint/validation_summary.json`；SHA-256：`f13dda8521860fc9fc0817cbbc62166f92e9068554341b9e536c0c7b77f708ef`
- 输入：`/content/ramen_semantic_v5/outputs_full/eval_sequential/metrics.json`；SHA-256：`f264cfe0b4aebd263aa79760e5a54555e0e0e0dd5305103c6c74aae2723ee3df`
- 输入：`/content/ramen_semantic_v5/outputs_full/sequential/validation_summary.json`；SHA-256：`bc348d28656550945edc3ef9554682f3aba3e7a42bf0dce482aa13a2bcbc538e`
- 输入：`/content/ramen_semantic_v5/outputs_full/eval_joint_descriptor_bank/metrics.json`；SHA-256：`d00fa820b6485a1ab45fec387f656f36bd694241498d7ff5c5c46574edc6c607`
- 输入：`/content/ramen_semantic_v5/outputs_full/monitored_state.json`；SHA-256：`183e85abeed2fa23b4be8870781f689eb5747820329beb3b2a72088df43b576a`

