import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_ramen_report import build_evidence, main, markdown_report


class RamenReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.report = self.root / "report"

    def write(self, relative, payload):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def metrics(psnr=30.0):
        return {"iteration": 15000, "test_psnr": psnr, "test_ssim": 0.9,
                "mean_iou": 0.5, "mean_boundary_iou": 0.3, "gaussians": 100,
                "threshold": 0.25, "granularity": 1, "boundary_ratio": 0.008,
                "evaluator_version": "abc", "dataset_fingerprint": "data",
                "per_label_iou": {"apple": 0.5},
                "reconstruction_rows": [{"split": "0", "camera": "test_0", "psnr": psnr}],
                "rows": [{"split": "0", "camera": "test_0", "label": "apple", "iou": 0.5, "boundary_iou": 0.3}]}

    def evidence(self):
        return build_evidence(self.output, report_dir=self.report)

    def test_missing_evaluation_does_not_use_all_test_log_or_claim_finished(self):
        self.write("audit_notes.json", {"training_log_observations": [{
            "run": "joint", "metric": "test_psnr", "value": 30.279134,
            "scope": "all_test_cameras", "source": "train.log:20", "iteration": 15000}]})
        evidence = self.evidence()
        self.assertNotIn("test_psnr", evidence["current"]["runs"]["joint"]["values"])
        self.assertEqual(evidence["comparisons"]["joint_minus_sequential"]["deltas"], {})
        report = markdown_report(evidence)
        self.assertIn("最终对照尚未完整", report)
        self.assertIn("pending", report)
        self.assertIn("all_test_cameras", report)
        self.assertFalse(evidence["timing"]["certified"])

    def test_matching_evaluator_records_allow_raw_differences_and_preserve_zero(self):
        left = self.metrics(30.25)
        left["mean_iou"] = 0.0
        self.write("eval_joint/metrics.json", left)
        self.write("eval_sequential/metrics.json", self.metrics(30.0))
        evidence = self.evidence()
        comparison = evidence["comparisons"]["joint_minus_sequential"]
        self.assertEqual(comparison["status"], "recorded_protocol_matches")
        self.assertEqual(comparison["deltas"]["test_psnr"], 0.25)
        self.assertEqual(comparison["deltas"]["mean_iou"], -0.5)
        self.assertEqual(evidence["current"]["runs"]["joint"]["per_label_boundary_iou"]["apple"], 0.3)
        self.assertIn("0.000000", markdown_report(evidence))

    def test_different_views_or_parameters_prevent_improvement_claim(self):
        for changed in ("camera", "threshold", "dataset_fingerprint"):
            with self.subTest(changed=changed):
                left, right = self.metrics(31.0), self.metrics(30.0)
                if changed == "camera":
                    right["reconstruction_rows"][0]["camera"] = "test_1"
                else:
                    right[changed] = 0.3 if changed == "threshold" else "other_data"
                self.write("eval_joint/metrics.json", left)
                self.write("eval_sequential/metrics.json", right)
                result = self.evidence()["comparisons"]["joint_minus_sequential"]
                self.assertEqual(result["status"], "protocol_mismatch")
                self.assertEqual(result["deltas"], {})

    def test_unrecorded_boundary_ratio_does_not_allow_boundary_delta(self):
        metrics = self.metrics()
        del metrics["boundary_ratio"]
        self.write("eval_joint/metrics.json", metrics)
        self.write("eval_sequential/metrics.json", self.metrics())
        result = self.evidence()["comparisons"]["joint_minus_sequential"]
        self.assertIn("test_psnr", result["deltas"])
        self.assertNotIn("mean_boundary_iou", result["deltas"])

    def test_same_numbers_or_threshold_do_not_make_different_scoring_protocols_comparable(self):
        for changed in ({"score_mode": "clip_cosine"}, {"mask_metric_protocol": "gg_native"}, {"alpha_min": 0.01}):
            with self.subTest(changed=changed):
                left, right = self.metrics(), self.metrics()
                left["protocol"] = changed
                self.write("eval_joint/metrics.json", left)
                self.write("eval_sequential/metrics.json", right)
                result = self.evidence()["comparisons"]["joint_minus_sequential"]
                self.assertEqual(result["status"], "protocol_mismatch")
                self.assertEqual(result["deltas"], {})

    def test_requested_equal_time_without_complete_history_is_not_certified(self):
        self.write("comparison.json", {"protocol": {"equal_wall_clock": True}})
        self.write("training_times.json", {"joint_train_seconds": 100,
                   "sequential_rgb_seconds": 60, "sequential_semantic_seconds": 40})
        result = self.evidence()["timing"]
        self.assertTrue(result["requested"])
        self.assertFalse(result["certified"])
        self.assertIn("不能认证", result["status"])
        self.assertIsNone(result["sequential_seconds"])

    def test_complete_timing_has_explicit_tolerance(self):
        self.write("comparison.json", {"protocol": {"equal_wall_clock_requested": True}})
        timings = {}
        for name, duration in (("joint_train_seconds", 100), ("sequential_rgb_seconds", 60),
                               ("sequential_semantic_seconds", 41)):
            timings[name] = duration
            timings[name + "_timing_complete"] = True
            timings[name + "_completed"] = True
        self.write("training_times.json", timings)
        result = self.evidence()["timing"]
        self.assertTrue(result["certified"])
        self.assertEqual(result["delta_seconds"], 1)
        self.assertEqual(result["tolerance_seconds"], 5)
        timings["sequential_semantic_seconds"] = 60
        self.write("training_times.json", timings)
        self.assertFalse(self.evidence()["timing"]["certified"])

    def test_repeatable_output_does_not_modify_results_and_cannot_invent_background(self):
        self.write("comparison.json", {"important": ["apple"], "normal": ["cup"]})
        source = self.write("eval_joint/metrics.json", self.metrics())
        before = source.read_bytes()
        args = ["--output_root", str(self.output), "--report_dir", str(self.report)]
        main(args)
        first = {path.name: path.read_bytes() for path in self.report.iterdir()}
        main(args)
        second = {path.name: path.read_bytes() for path in self.report.iterdir()}
        self.assertEqual(first, second)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(set(first), {"ramen_final_report.md", "ramen_evidence.json"})
        self.assertIn("背景", first["ramen_final_report.md"].decode())
        self.assertIn("pending", first["ramen_final_report.md"].decode())

    def test_nonfinite_and_corrupt_inputs_remain_pending(self):
        bad = self.output / "comparison.json"
        bad.write_text("broken", encoding="utf-8")
        metrics = self.metrics()
        metrics["mean_iou"] = float("nan")
        self.write("eval_joint/metrics.json", metrics)
        evidence = self.evidence()
        self.assertNotIn("mean_iou", evidence["current"]["runs"]["joint"]["values"])
        self.assertTrue(evidence["warnings"])
        json.dumps(evidence, allow_nan=False)

    def bank_metrics(self):
        metrics = self.metrics(35.0)
        metrics.update(mean_iou=0.95, per_label_iou={"apple": 0.95})
        metrics["protocol"] = {
            "score_mode": "descriptor_bank", "mask_metric_protocol": "gg_native",
            "descriptor_bank": {"sha256": "bank-hash", "retrieval": {"text_threshold": 0.5},
                                "construction": {"elapsed_seconds": 12.0, "source_views": ["train_0"]}},
        }
        metrics["all_test_rgb"] = {"metric_scope": "all_test_cameras", "camera_count": 4,
                                   "psnr": 36.0, "ssim": 0.98}
        return metrics

    def test_bank_is_independent_and_does_not_change_main_metrics_deltas_or_time(self):
        self.write("eval_joint/metrics.json", self.metrics(30.25))
        self.write("eval_sequential/metrics.json", self.metrics(30.0))
        self.write("comparison.json", {"protocol": {"equal_wall_clock_requested": True}})
        timings = {}
        for name, seconds in (("joint_train_seconds", 100), ("sequential_rgb_seconds", 60),
                              ("sequential_semantic_seconds", 40)):
            timings.update({name: seconds, name + "_completed": True, name + "_timing_complete": True})
        self.write("training_times.json", timings)
        before = self.evidence()
        source = self.write("eval_joint_descriptor_bank/metrics.json", self.bank_metrics())
        self.write("monitored_state.json", {"status": "running", "stage": "report",
                   "computation_complete": False, "full_final_archive_completed": False,
                   "stages": {"descriptor_bank": {"returncode": 0, "observed_wrapper_seconds": 9000},
                              "descriptor_evaluation": {"returncode": 0, "observed_wrapper_seconds": 5000}}})
        after = self.evidence()
        for key in ("current", "legacy", "comparisons", "timing"):
            self.assertEqual(after[key], before[key])
        bank = after["additional_evaluations"]["joint_descriptor_bank"]
        self.assertTrue(bank["excluded_from_main_comparison"])
        self.assertTrue(bank["excluded_from_main_equal_time_certification"])
        self.assertEqual(bank["values"]["test_psnr"], 35.0)
        self.assertEqual(bank["all_test_rgb"]["psnr"], 36.0)
        self.assertEqual(bank["per_label_boundary_iou"]["apple"], 0.3)
        self.assertIsNone(bank["costs"]["end_to_end_seconds"])
        self.assertEqual(bank["costs"]["builder_recorded_seconds"], 12.0)
        self.assertIn(str(source.resolve()), [item["path"] for item in after["inputs"]])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in after["inputs"]))
        report = markdown_report(after)
        self.assertIn("独立附表：联合模型＋训练后描述符库", report)
        self.assertIn("不纳入主等时间比较", report)
        self.assertIn("不是纯 GPU 时间", report)
        self.assertIn("最终 mask 阈值与候选 CLIP 文本门限含义不同", report)
        self.assertIn("完整端到端额外成本 | pending", report)
        self.assertFalse(bank["monitor_snapshot"]["computation_complete"])

    def test_missing_or_wrong_bank_protocol_does_not_reuse_main_result(self):
        self.write("eval_joint/metrics.json", self.metrics())
        missing = self.evidence()["additional_evaluations"]["joint_descriptor_bank"]
        self.assertEqual(missing["status"], "pending")
        self.assertEqual(missing["values"], {})
        self.write("eval_joint_descriptor_bank/metrics.json", self.metrics(99.0))
        evidence = self.evidence()
        bank = evidence["additional_evaluations"]["joint_descriptor_bank"]
        self.assertEqual(bank["status"], "wrong_protocol")
        self.assertEqual(bank["values"], {})
        self.assertNotIn("99.000000", markdown_report(evidence))
        self.assertTrue(evidence["warnings"])

    def test_incomplete_bank_cost_or_unknown_all_test_scope_stays_pending(self):
        metrics = self.bank_metrics()
        metrics["protocol"]["descriptor_bank"]["construction"]["elapsed_seconds"] = float("nan")
        metrics["all_test_rgb"].pop("metric_scope")
        self.write("outputs_full/eval_joint_descriptor_bank/metrics.json", metrics)
        self.write("monitored_state.json", {"stages": {
            "descriptor_bank": {"returncode": 1, "observed_wrapper_seconds": 60},
            "descriptor_evaluation": {"observed_wrapper_seconds": 20}}})
        evidence = self.evidence()
        bank = evidence["additional_evaluations"]["joint_descriptor_bank"]
        self.assertEqual(bank["all_test_rgb"], {})
        self.assertIsNone(bank["costs"]["builder_recorded_seconds"])
        for name in ("descriptor_bank", "descriptor_evaluation"):
            self.assertIsNone(bank["costs"][name]["observed_wrapper_seconds"])
        json.dumps(evidence, allow_nan=False)

    def test_bank_report_regeneration_preserves_all_source_files(self):
        metrics = self.write("eval_joint_descriptor_bank/metrics.json", self.bank_metrics())
        state = self.write("monitored_state.json", {"status": "complete", "stages": {}})
        saved = {path: path.read_bytes() for path in (metrics, state)}
        args = ["--output_root", str(self.output), "--report_dir", str(self.report)]
        main(args)
        first = {path.name: path.read_bytes() for path in self.report.iterdir()}
        main(args)
        self.assertEqual(first, {path.name: path.read_bytes() for path in self.report.iterdir()})
        self.assertEqual(saved, {path: path.read_bytes() for path in saved})
        self.assertEqual(set(first), {"ramen_final_report.md", "ramen_evidence.json"})


if __name__ == "__main__":
    unittest.main()
