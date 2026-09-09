"""Build an evidence-only Chinese Ramen report, without loading models or a GPU.

Example:
    python -m scripts.build_ramen_report --output_root /run/outputs_full \
        --legacy_root /run/legacy_metrics_backup --report_dir /run/report

Only ramen_final_report.md and ramen_evidence.json are written. Missing values
stay pending; a training-log PSNR is never substituted for annotated evaluation.
Optional audit_notes.json and cleanup_manifest.json may be placed in output_root
or its parent (or report_dir). Audit notes can contain ``notes`` and
``training_log_observations``; a log observation needs metric/value/scope/source.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = (
    ("test_psnr", "PSNR (dB) ↑"),
    ("test_ssim", "SSIM ↑"),
    ("mean_iou", "mIoU ↑"),
    ("mean_boundary_iou", "Boundary-IoU ↑"),
    ("gaussians", "高斯数量"),
)
TIERS = (("important", "重要物品"), ("normal", "普通物品"),
         ("background", "背景"))
RUNS = (("joint", "本次联合模型"), ("sequential", "本次顺序基线"))
PENDING = "pending（未取得证据）"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def display(value, signed=False):
    if number(value):
        if isinstance(value, int):
            return f"{value:+,}" if signed else f"{value:,}"
        return f"{value:+.6f}" if signed else f"{value:.6f}"
    return PENDING


def cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


class EvidenceReader:
    def __init__(self):
        self.inputs = []
        self.warnings = []

    def read(self, candidates):
        for path in candidates:
            if not path.is_file():
                continue
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("JSON root must be an object")
            except (OSError, ValueError) as error:
                self.warnings.append(f"无法读取 {path}: {error}")
                continue
            source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw)}
            if source not in self.inputs:
                self.inputs.append(source)
            return clean_json(value), source["path"]
        return {}, None


def collect_run(root, reader):
    if root is None:
        return {"root": None, "comparison": {}, "runs": {}, "timings": {}}
    root = Path(root).resolve()
    bases = [root, root / "outputs_full"]
    comparison, comparison_source = reader.read([base / "comparison.json" for base in bases])
    timings, timing_source = reader.read([base / "training_times.json" for base in bases])
    if not timings:
        timings = comparison.get("protocol", {}).get("timings_seconds", {})
        timing_source = comparison_source if timings else None
    result = {"root": str(root), "comparison": comparison, "timings": timings,
              "timing_source": timing_source, "comparison_source": comparison_source, "runs": {}}
    for name, _ in RUNS:
        raw, source = reader.read([base / f"eval_{name}" / "metrics.json" for base in bases])
        summary = comparison.get(name, {})
        if not isinstance(summary, dict):
            summary = {}
        # The raw evaluator wins, but conflicting copied summary data are exposed.
        conflicts = [key for key, _ in METRICS if number(raw.get(key))
                     and number(summary.get(key)) and raw[key] != summary[key]]
        if conflicts:
            reader.warnings.append(f"{name} 原始评估与 comparison.json 不一致：{', '.join(conflicts)}；采用原始评估。")
        merged = {**summary, **raw}
        values = {key: value for key, value in merged.items() if number(value)}
        validation, validation_source = reader.read(
            [base / name / "validation_summary.json" for base in bases])
        if not validation:
            validation = summary.get("validation", {})
            validation_source = comparison_source if validation else None
        per_label = merged.get("per_label_iou", {})
        boundary = merged.get("per_label_boundary_iou", {})
        if not isinstance(boundary, dict):
            boundary = {}
        else:
            boundary = dict(boundary)
        rows = raw.get("rows", [])
        if isinstance(rows, list):
            labels = sorted({row["label"] for row in rows if isinstance(row, dict)
                             and isinstance(row.get("label"), str)})
            for label in labels:
                values_for_label = [row["boundary_iou"] for row in rows
                                    if isinstance(row, dict) and row.get("label") == label
                                    and number(row.get("boundary_iou"))]
                if label not in boundary and values_for_label:
                    boundary[label] = sum(values_for_label) / len(values_for_label)
        result["runs"][name] = {
            "values": values,
            "per_label_iou": {key: value for key, value in per_label.items() if number(value)}
            if isinstance(per_label, dict) else {},
            "per_label_boundary_iou": {key: value for key, value in boundary.items() if number(value)},
            "tier_gaussians": merged.get("tier_gaussians", {}),
            "validation": validation if isinstance(validation, dict) else {},
            "validation_source": validation_source,
            "metric_source": source or comparison_source,
            "metric_origin": "annotated_evaluator" if source else "comparison_summary" if summary else "missing",
            "protocol": evaluation_protocol(raw),
        }
    return result


def evaluation_protocol(raw):
    """Keep camera identity, not a count: four log views are not three mask views."""
    protocol = dict(raw.get("protocol", {})) if isinstance(raw.get("protocol"), dict) else {}
    for key in ("threshold", "granularity", "boundary_ratio", "evaluator_version", "dataset_fingerprint"):
        if key in raw:
            protocol[key] = raw[key]
    reconstruction_rows = raw.get("reconstruction_rows", [])
    mask_rows = raw.get("rows", [])
    protocol["reconstruction_views"] = sorted([
        [str(row.get("split", "")), str(row["camera"])]
        for row in reconstruction_rows if isinstance(row, dict) and row.get("camera") is not None
    ]) if isinstance(reconstruction_rows, list) else []
    protocol["mask_views_and_labels"] = sorted([
        [str(row.get("split", "")), str(row["camera"]), str(row["label"])]
        for row in mask_rows if isinstance(row, dict) and row.get("camera") is not None
        and row.get("label") is not None
    ]) if isinstance(mask_rows, list) else []
    return protocol


def compare_runs(left, right):
    """left - right only after matching the recorded annotated-view protocol."""
    left, right = left or {}, right or {}
    lp, rp = left.get("protocol", {}), right.get("protocol", {})
    missing, mismatch = [], []
    if any(item.get("metric_origin") != "annotated_evaluator" for item in (left, right)):
        missing.append("双方原始 annotated evaluator metrics.json")
    legacy_defaults = {
        "score_mode": "legacy_pca_cosine", "mask_metric_protocol": "legacy",
        "score_compositing": "clipped_per_gaussian_cosine_then_render",
        "score_space": "centered_pca", "alpha_min": 0.0,
        "boundary_pad_edges": False, "iou_aggregation": "mean_over_view_label_rows",
        "layer_selection": "fixed_explicit_granularity_no_test_selection",
    }
    for key, default in legacy_defaults.items():
        if lp.get(key, default) != rp.get(key, default):
            mismatch.append(key)
    for key in ("negative_prompts", "relevancy_temperature"):
        if lp.get("score_mode") == "clip_relevancy" or rp.get("score_mode") == "clip_relevancy":
            if lp.get(key) is None or rp.get(key) is None:
                missing.append(key)
            elif lp[key] != rp[key]:
                mismatch.append(key)
    for key in ("reconstruction_views", "mask_views_and_labels", "threshold", "granularity"):
        if lp.get(key) is None or rp.get(key) is None or lp.get(key) == [] or rp.get(key) == []:
            missing.append(key)
        elif lp[key] != rp[key]:
            mismatch.append(key)
    for key in ("boundary_ratio", "evaluator_version", "dataset_fingerprint"):
        if key in lp and key in rp and lp[key] != rp[key]:
            mismatch.append(key)
    compatible = not missing and not mismatch
    deltas = {}
    if compatible:
        for key, _ in METRICS:
            if key == "mean_boundary_iou" and (lp.get("boundary_ratio") is None or rp.get("boundary_ratio") is None):
                continue
            lv, rv = left.get("values", {}).get(key), right.get("values", {}).get(key)
            if number(lv) and number(rv):
                deltas[key] = lv - rv
    cautions = []
    if lp.get("boundary_ratio") is None or rp.get("boundary_ratio") is None:
        cautions.append("Boundary-IoU 的 boundary_ratio 未完整记录，禁止计算该项改变量。")
    if lp.get("evaluator_version") is None or rp.get("evaluator_version") is None:
        cautions.append("评估器版本未完整记录；匹配仅指已记录的视角和参数，不能证明实现完全相同。")
    if lp.get("dataset_fingerprint") is None or rp.get("dataset_fingerprint") is None:
        cautions.append("数据/标注文件指纹未完整记录，不能认证不同运行的数据内容完全相同。")
    return {"status": "recorded_protocol_matches" if compatible else "protocol_mismatch" if mismatch else "pending",
            "missing": missing, "mismatch": mismatch, "cautions": cautions,
            "deltas": deltas, "scope": "annotated_evaluator_only"}


def timing_assessment(snapshot):
    timings = snapshot.get("timings", {})
    protocol = snapshot.get("comparison", {}).get("protocol", {})
    requested = bool(protocol.get("equal_wall_clock_requested", protocol.get("equal_wall_clock", False)))
    names = ("joint_train_seconds", "sequential_rgb_seconds", "sequential_semantic_seconds")
    missing = [name for name in names if not (
        number(timings.get(name)) and timings[name] > 0
        and timings.get(name + "_timing_complete") is True
        and timings.get(name + "_completed") is True
    )]
    total = sum(timings[name] for name in names[1:]) if not missing else None
    delta = total - timings[names[0]] if not missing else None
    tolerance = max(5.0, timings[names[0]] * 0.02) if not missing else None
    certified = bool(requested and not missing and abs(delta) <= tolerance)
    return {"requested": requested, "certified": certified, "missing": missing,
            "joint_seconds": timings.get(names[0]), "sequential_seconds": total,
            "delta_seconds": delta, "tolerance_seconds": tolerance,
            "status": "等时间已由完整阶段计时认证" if certified else "不能认证等时间对比",
            "reason": "缺失完整且已结束的阶段计时" if missing else
            "未声明等时间协议" if not requested else "实测用时超出容差" if not certified else
            "三个阶段完整计时，差值处于 max(5秒, 联合模型时长的2%) 内"}


def tier_values(snapshot, name, tier):
    run = snapshot.get("runs", {}).get(name, {})
    values = run.get("values", {})
    labels = snapshot.get("comparison", {}).get(tier)
    labels = labels if isinstance(labels, list) else []
    per_label = run.get("per_label_iou", {})
    mean_iou = values.get(f"{tier}_mean_iou")
    if labels and all(number(per_label.get(label)) for label in labels):
        mean_iou = sum(per_label[label] for label in labels) / len(labels)
    counts = run.get("tier_gaussians", {})
    return {"labels": labels, "psnr": values.get(f"test_{tier}_psnr"), "mean_iou": mean_iou,
            "gaussians": counts.get(tier) if isinstance(counts, dict) else None}


def build_evidence(output_root, legacy_root=None, report_dir=None):
    reader = EvidenceReader()
    root = Path(output_root).resolve()
    report_dir = Path(report_dir or root / "report").resolve()
    current = collect_run(root, reader)
    legacy = collect_run(legacy_root, reader)
    optional_bases = list(dict.fromkeys([root, root.parent, report_dir]))
    audit, _ = reader.read([base / "audit_notes.json" for base in optional_bases])
    cleanup, _ = reader.read([base / "cleanup_manifest.json" for base in optional_bases])
    observations = []
    for observation in audit.get("training_log_observations", []):
        if isinstance(observation, dict) and number(observation.get("value")) and all(
            observation.get(key) for key in ("metric", "scope", "source")
        ):
            observations.append(observation)
        else:
            reader.warnings.append("一条训练日志观察缺少数值、metric、scope 或 source，未纳入指标表。")
    return {"schema_version": 1, "current": current, "legacy": legacy,
            "comparisons": {
                "joint_minus_sequential": compare_runs(current["runs"].get("joint"), current["runs"].get("sequential")),
                "joint_minus_legacy_joint": compare_runs(current["runs"].get("joint"), legacy["runs"].get("joint")),
            }, "timing": timing_assessment(current), "audit_notes": audit,
            "training_log_observations": observations, "cleanup_manifest": cleanup,
            "inputs": reader.inputs, "warnings": reader.warnings,
            "artifacts": {"markdown": str(report_dir / "ramen_final_report.md"),
                          "evidence_json": str(report_dir / "ramen_evidence.json")}}


def markdown_report(evidence):
    current, legacy = evidence["current"], evidence["legacy"]
    joint = current["runs"].get("joint", {})
    baseline = current["runs"].get("sequential", {})
    old_joint = legacy["runs"].get("joint", {})
    lines = ["# Ramen 联合重建与语义分割实验报告", "",
             "本报告仅汇总已读取的实验文件。pending 表示证据尚缺，不代表零分、失败或无提升。", "",
             "## 实验结论与完成状态", ""]
    complete = all(run.get("metric_origin") == "annotated_evaluator" and all(
        number(run.get("values", {}).get(key)) for key, _ in METRICS)
        for run in (joint, baseline))
    lines.append("双方原始评估包含所列核心指标；具体结论仍受下述协议与审计限制。" if complete else
                 "最终对照尚未完整：至少一方缺少原始评估或核心指标，不能宣称全部实验完成。")
    lines.extend(["", "单次 PSNR 微小变化不能证明稳定提升；高斯数量下降也不能单独证明算法更有效。", "",
                  "## 标注视角评估结果（协议核验见下文）", "",
                  "以下数值来自 eval_*/metrics.json，或明确标记的 comparison.json 摘要；只覆盖该评估器记录的标注视角。", "",
                  "| 指标 | 本次联合模型 | 本次顺序基线 | 历史联合模型 |", "|---|---:|---:|---:|"])
    for key, title in METRICS:
        lines.append("| " + " | ".join([title] + [display(run.get("values", {}).get(key))
                                                      for run in (joint, baseline, old_joint)]) + " |")
    lines.extend(["", "| 运行 | 指标来源类型 | 标注重建视角数 | 迭代标签 |", "|---|---|---:|---:|"])
    for title, run in (("本次联合", joint), ("本次顺序", baseline), ("历史联合", old_joint)):
        views = run.get("protocol", {}).get("reconstruction_views", [])
        lines.append(f"| {title} | {run.get('metric_origin', 'missing')} | {len(views) if views else PENDING} | {display(run.get('values', {}).get('iteration'))} |")
    lines.extend(["", "### 对比资格与改变量", ""])
    for key, title in (("joint_minus_sequential", "本次联合 − 本次顺序"),
                       ("joint_minus_legacy_joint", "本次联合 − 历史联合")):
        result = evidence["comparisons"][key]
        lines.append(f"- {title}：{result['status']}。")
        if result["missing"]:
            lines.append("  缺少：" + cell(", ".join(result["missing"])) + "。")
        if result["mismatch"]:
            lines.append("  协议不同：" + cell(", ".join(result["mismatch"])) + "，不计算提升。")
        if result["deltas"]:
            lines.append("  已记录口径下的原始差值：" + "；".join(
                f"{label} {display(result['deltas'][metric], signed=True)}" for metric, label in METRICS
                if metric in result["deltas"]) + "。这不是统计显著性结论。")
        for caution in result["cautions"]:
            lines.append("  " + caution)
    lines.extend(["", "## 训练日志与验证集（独立口径）", "",
                  "训练日志 all-test PSNR 与上述 annotated evaluator PSNR 不混用，不跨口径相减。", "",
                  "| 运行 | 日志指标 | 数值 | 迭代 | 范围 | 证据来源 |", "|---|---|---:|---:|---|---|"])
    if not evidence["training_log_observations"]:
        lines.append(f"| — | {PENDING} | — | — | — | — |")
    for observation in evidence["training_log_observations"]:
        lines.append("| " + " | ".join([cell(observation.get("run", "未标明")), cell(observation["metric"]),
                    display(observation["value"]), display(observation.get("iteration")),
                    cell(observation["scope"]), cell(observation["source"])]) + " |")
    lines.extend(["", "| 运行 | 记录的最佳验证 PSNR | 最佳迭代 | 实际训练迭代 | 导出迭代别名 |", "|---|---:|---:|---:|---:|"])
    for title, run in (("本次联合", joint), ("本次顺序", baseline), ("历史联合", old_joint)):
        validation = run.get("validation", {})
        lines.append("| " + " | ".join([title] + [display(validation.get(key)) for key in
                    ("best_psnr", "best_iteration", "trained_iterations", "selected_iteration_alias")]) + " |")
    lines.extend(["", "best_psnr / best_iteration 是训练程序记录的选择结果，可能受 min_delta 约束，不一定等于 history 数学最大值。最佳 checkpoint、末轮 checkpoint 和导出别名必须区分；以上表格不把不同 checkpoint 的分数合并。", "",
                  "## 三级重要性与逐类别结果", "",
                  "| 模型 | 级别 | 该级标签 | 区域 PSNR | 类别平均 IoU | 高斯数量 |", "|---|---|---|---:|---:|---:|"])
    for run_name, title in RUNS:
        for tier, tier_title in TIERS:
            value = tier_values(current, run_name, tier)
            lines.append("| " + " | ".join([title, tier_title, cell(", ".join(value["labels"])) or PENDING,
                        display(value["psnr"]), display(value["mean_iou"]), display(value["gaussians"])]) + " |")
    lines.extend(["", "背景是剩余区域，不自动等同于一个有完整人工标注的语义类别；不以 1 − 前景 IoU 推算背景 IoU。", "",
                  "| 类别 | 联合 IoU | 顺序 IoU | 历史联合 IoU | 联合 Boundary-IoU | 顺序 Boundary-IoU |", "|---|---:|---:|---:|---:|---:|"])
    labels = sorted(set().union(*(set(run.get("per_label_iou", {})) for run in (joint, baseline, old_joint))))
    for label in labels:
        values = [display(run.get("per_label_iou", {}).get(label)) for run in (joint, baseline, old_joint)]
        values += [display(run.get("per_label_boundary_iou", {}).get(label)) for run in (joint, baseline)]
        lines.append("| " + " | ".join([cell(label)] + values) + " |")
    if not labels:
        lines.append(f"| {PENDING} | — | — | — | — | — |")
    lines.extend(["", "## 训练用时与公平性", "",
                  evidence["timing"]["status"] + "。" + evidence["timing"]["reason"] + "。", "",
                  f"联合训练：{display(evidence['timing']['joint_seconds'])} 秒；顺序 RGB＋语义：{display(evidence['timing']['sequential_seconds'])} 秒；差值：{display(evidence['timing']['delta_seconds'], signed=True)} 秒。", "",
                  "断点续训前的时长缺失、阶段未完成或仅传入 --equal_time 均不能认证等时间。预处理、评估、GPU型号与重复试验耗时需另外记录，阶段用时不自动等于端到端成本。", "",
                  "## 审计限制与已知问题", ""])
    notes = evidence["audit_notes"].get("notes", [])
    if isinstance(notes, (str, dict)):
        notes = [notes]
    for note in notes:
        lines.append("- " + cell(note))
    for warning in evidence["warnings"]:
        lines.append("- " + cell(warning))
    lines.extend(["- 单场景、单随机种子的结果不能直接代表稀疏视角、单图补全或其他场景能力。",
                  "- 伪标签训练损失下降不等同于人工标注 mIoU 提升；边界与跨视角指标需要各自评估。",
                  "- 验证集必须与 PCA / 聚类 / 原型构建等学习步骤隔离；若共享拟合，则须标明 transductive，不能作为完全独立验证。", "",
                  "## 可执行改进建议", "",
                  "1. 先修复并验证正确性：对重要性投影、遮挡与像素坐标做可视化叠加；核对 CUDA 实际支持的 SH 阶数，避免无效参数占用显存。",
                  "2. 固定相同的人工标注测试视角、图像分辨率、mask 文件、阈值、粒度、boundary_ratio 与评估代码版本，补齐联合和顺序模型的同口径评估。",
                  "3. 将预处理拟合限制到训练视角，并冻结验证/测试变换。对 RGB 预热 0 / 1500 / 3000 轮做验证集选择；测试集仅用于最终一次确认。",
                  "4. 单独消融边界加权与细长物体致密化：保持可比高斯预算，记录每类 IoU、Boundary-IoU、重要物体区域 PSNR，以及边缘误删/漏检图。",
                  "5. 检查跨视角正样本的遮挡和置信度，剔除不可靠对应；分别消融跨视角项与语义项，检验是否改善人工标注而非仅降低伪标签损失。",
                  "6. 修复后至少运行多个随机种子，报告均值与离散度；在同一GPU记录完整训练阶段累计时间，再开展等时间对照。", "",
                  "上述为待验证改进方案，不代表已经完成训练或取得提升。", "",
                  "## 云盘清理记录", ""])
    cleanup = evidence["cleanup_manifest"]
    if cleanup:
        items = cleanup.get("items", [])
        lines.extend(["| 路径 | 动作 | 字节 | 可恢复性 | 原因 |", "|---|---|---:|---|---|"])
        for item in items:
            if isinstance(item, dict):
                lines.append("| " + " | ".join([cell(item.get("path", "未记录")), cell(item.get("action", "未记录")),
                    display(item.get("bytes")), cell(item.get("recoverable", "未记录")), cell(item.get("reason", "未记录"))]) + " |")
        if not items:
            lines.append("| 详细记录见证据 JSON | — | — | — | — |")
        lines.append("\n清理记录是已提供的审计信息；本报告脚本不会执行删除，也不会推断操作成功。")
    else:
        lines.append(PENDING + "；未取得清理清单，不能声称已释放云盘空间。")
    lines.extend(["", "## 产物与原始证据", ""])
    for name, path in evidence["artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    for source in evidence["inputs"]:
        lines.append(f"- 输入：`{source['path']}`；SHA-256：`{source['sha256']}`")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成有来源、缺失显式标记的中文 Ramen 实验报告")
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--legacy_root", type=Path)
    parser.add_argument("--report_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    evidence = build_evidence(args.output_root, args.legacy_root, args.report_dir)
    inputs = {item["path"] for item in evidence["inputs"]}
    if inputs.intersection(evidence["artifacts"].values()):
        parser.error("报告输出路径不得覆盖输入证据")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    Path(evidence["artifacts"]["markdown"]).write_text(markdown_report(evidence), encoding="utf-8")
    Path(evidence["artifacts"]["evidence_json"]).write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence["artifacts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
