"""Notebook contract/backup-selection tests; never execute Colab/GPU cells."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "colab_ramen_semantic_v5.ipynb"


class V5NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())
        cls.cells = {cell["id"]: "".join(cell["source"]) for cell in cls.notebook["cells"]}

    def function(self, cell, name):
        node = next(node for node in ast.walk(ast.parse(self.cells[cell])) if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {"Path": Path, "json": json, "complete_torch_archive": lambda path: Path(path).is_file() and Path(path).read_bytes() != b"corrupt"}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_function", "exec"), namespace)
        return namespace[name]

    def assignment(self, cell, name, namespace):
        node = next(node for node in ast.walk(ast.parse(self.cells[cell])) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        return eval(compile(ast.Expression(node.value), "notebook_assignment", "eval"), namespace)

    def test_json_cells_compile_and_have_no_claimed_execution(self):
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def commands(self):
        namespace = {"sys": SimpleNamespace(executable="python"), "SCENE": Path("/content/full"), "SAM": Path("/content/sam"), "OUTPUT": Path("/content/full_output")}
        full = self.assignment("sam-cache", "BENCH", namespace)
        namespace.update(BENCH=full, SMOKE_SCENE=Path("/content/subset"), SMOKE_TEACHER_OUTPUT=Path("/content/subset_teacher"), replace_flag=self.function("prepare", "replace_flag"))
        smoke = self.assignment("prepare", "SMOKE_BENCH", namespace)
        return full, smoke

    def test_full_and_smoke_have_distinct_teachers_and_cropping(self):
        full, smoke = self.commands()
        option = lambda command, flag: command[command.index(flag) + 1]
        self.assertNotEqual(option(full, "--scene"), option(smoke, "--scene"))
        self.assertNotEqual(option(full, "--output_root"), option(smoke, "--output_root"))
        self.assertEqual(option(full, "--validation_views"), "12")
        self.assertEqual(option(smoke, "--validation_views"), "2")
        self.assertEqual(option(full, "--sam_crop_n_layers"), "1")
        self.assertEqual(option(smoke, "--sam_crop_n_layers"), "0")

    def test_formal_schedule_is_15k_with_original_initialization(self):
        full, _ = self.commands()
        option = lambda flag: full[full.index(flag) + 1]
        self.assertEqual(option("--iterations"), "15000")
        self.assertEqual(option("--semantic_start"), "2500")
        self.assertEqual(option("--semantic_ramp_iterations"), "2000")
        self.assertNotIn("--init_rgb_ply", full)
        self.assertNotIn("--no_equal_time", full)
        self.assertIn("'--resume','--skip_preprocess'", self.cells["train-full"])
        self.assertNotIn("disk_usage(PERSIST)", self.cells["train-full"])

    def test_smoke_is_300_steps_and_disables_geometry_reset(self):
        namespace = {"sys": SimpleNamespace(executable="python"), "SMOKE_SCENE": Path("/content/subset"), "SMOKE": Path("/content/smoke")}
        command = self.assignment("smoke", "smoke_command", namespace)
        option = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(option("--iterations"), "300")
        self.assertEqual(option("--semantic_start"), "0")
        self.assertEqual(option("--semantic_ramp_iterations"), "100")
        self.assertEqual(option("--densify_until_iter"), "0")
        self.assertGreater(int(option("--opacity_reset_interval")), 300)
        self.assertIn("smoke_protocol", self.cells["smoke"])
        self.assertIn("complete_torch_archive(p)", self.cells["smoke"])

    def backup_fixture(self, root):
        root = Path(root)
        scene, output = root / "data/ramen", root / "outputs_full"
        def write(relative, value="fixture"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
            return path
        for name in ("semantic_meta.npz", "semantic_summary.json", "scene_inventory.json", "sparse/0/train.txt", "sparse/0/val.txt", "sparse/0/test.txt"):
            write("data/ramen/" + name)
        for directory in ("semantic_maps", "importance_masks", "semantic_raw", "detail_weights"):
            write("data/ramen/" + directory + "/frame.npz")
        for name in ("v5_teacher_files.json", "experiment_protocol.json"):
            write("outputs_full/" + name)
        write("outputs_full/joint/cfg_args")
        return scene, output, write

    def test_partial_backup_retains_optimizer_markers_diagnostics_and_teacher(self):
        choose = self.function("selected-backup", "select_backup_files")
        with tempfile.TemporaryDirectory() as directory:
            scene, output, write = self.backup_fixture(directory)
            write("outputs_full/joint/chkpnt7000.pth")
            write("outputs_full/joint/chkpnt10000.pth", "corrupt")
            write("outputs_full/joint/semantic_training_diagnostics.jsonl")
            write("outputs_full/joint/semantic/iteration_15000/semantic_checkpoint.pt")
            write("outputs_full/joint/semantic/iteration_15000/training_complete.json")
            selected, _ = choose(Path(directory), scene, output, "resume")
            self.assertIn("outputs_full/joint/chkpnt7000.pth", selected)
            self.assertNotIn("outputs_full/joint/chkpnt10000.pth", selected)
            self.assertIn("outputs_full/joint/semantic_training_diagnostics.jsonl", selected)
            self.assertIn("outputs_full/joint/semantic/iteration_15000/semantic_checkpoint.pt", selected)
            self.assertIn("outputs_full/joint/semantic/iteration_15000/training_complete.json", selected)
            self.assertIn("data/ramen/detail_weights/frame.npz", selected)
            self.assertIn("data/ramen/semantic_maps/frame.npz", selected)

    def test_best_only_resume_emits_local_numeric_alias_for_runner(self):
        choose = self.function("selected-backup", "select_backup_files")
        with tempfile.TemporaryDirectory() as directory:
            scene, output, write = self.backup_fixture(directory)
            write("outputs_full/joint/best_val_chkpnt.pth")
            write("outputs_full/joint/validation_summary.json", json.dumps({"best_iteration": 6000}))
            selected, aliases = choose(Path(directory), scene, output, "resume")
            self.assertIn("outputs_full/joint/best_val_chkpnt.pth", selected)
            self.assertEqual(aliases[0]["target"], "outputs_full/joint/chkpnt6000.pth")

    def test_final_backup_cannot_claim_completion_without_comparison(self):
        choose = self.function("selected-backup", "select_backup_files")
        with tempfile.TemporaryDirectory() as directory:
            scene, output, _ = self.backup_fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "最终对比"):
                choose(Path(directory), scene, output, "final")

    def test_backup_and_restore_are_disabled_by_default(self):
        self.assertIs(self.assignment("selected-backup", "RUN_SELECTED_BACKUP", {}), False)
        self.assertEqual(self.assignment("restore-selected", "RESTORE_FROM", {}), "")
        self.assertIn("sha256(target)==metadata['sha256']", self.cells["selected-backup"])

    def test_restore_rejects_traversal_and_absolute_paths(self):
        safe = self.function("restore-selected", "safe_relative")
        for path in ("../outside", "/tmp/outside", ".", "data/../../outside"):
            with self.assertRaises(AssertionError):
                safe(path)
        self.assertEqual(safe("outputs_full/joint/chkpnt7000.pth"), Path("outputs_full/joint/chkpnt7000.pth"))


if __name__ == "__main__":
    unittest.main()
