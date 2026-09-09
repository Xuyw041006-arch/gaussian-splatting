"""Process/session contracts with fake children; no GPU or Drive access."""

import contextlib
import fcntl
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import run_v5_monitored_benchmark as session


class FakeChild:
    def __init__(self, returncode=0):
        self.pid = 7654321
        self.returncode = returncode
        self.polls = 0

    def poll(self):
        self.polls += 1
        return None if self.polls == 1 else self.returncode

    def wait(self, timeout=None):
        return self.returncode


class MonitoredSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.scene, self.output, self.snapshot = root / "scene", root / "output", root / "snapshot"
        self.scene.mkdir()
        self.output.mkdir()
        self.sam = root / "sam.pth"
        self.sam.touch()
        (self.output / "v5_teacher_files.json").write_text("{}")
        self.argv = ["monitored", "--scene", str(self.scene), "--output_root", str(self.output),
                     "--sam_checkpoint", str(self.sam), "--snapshot_dir", str(self.snapshot)]
        self.spawned = []
        self.fail_child = None
        self.create_comparison = True
        self.snapshot_calls = 0
        self.signal_handlers = {}

    def complete_snapshot(self, *_args, **_kwargs):
        self.snapshot_calls += 1
        payload = {"state": "snapshot_complete", "teacher": {"state": "complete"}, "checkpoints": {}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    def child(self, command, **options):
        self.spawned.append((command, options))
        stage = next((part for part in command if str(part).startswith("scripts.")), "unknown")
        if stage == "scripts.run_ramen_benchmark" and self.create_comparison:
            (self.output / "comparison.json").write_text(json.dumps({"protocol": {"equal_wall_clock": False}}))
        if stage == "scripts.build_semantic_descriptor_bank":
            bank = Path(command[command.index("--output") + 1])
            bank.parent.mkdir(parents=True, exist_ok=True)
            bank.write_bytes(b"fake bank")
        if stage == "scripts.evaluate_lerf_mask":
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "metrics.json").write_text("{}")
        if stage == "scripts.build_ramen_report":
            output = Path(command[command.index("--report_dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "ramen_final_report.md").write_text("fake report")
        return FakeChild(1 if stage == self.fail_child else 0)

    @contextlib.contextmanager
    def environment(self, snapshot=None, sleep=None):
        def install_signal(number, handler):
            previous = self.signal_handlers.get(number, signal.SIG_DFL)
            self.signal_handlers[number] = handler
            return previous

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(session.sys, "argv", self.argv))
            stack.enter_context(mock.patch.object(session.subprocess, "check_output", return_value="abc123\n"))
            stack.enter_context(mock.patch.object(session.subprocess, "run", side_effect=snapshot or self.complete_snapshot))
            stack.enter_context(mock.patch.object(session.subprocess, "Popen", side_effect=self.child))
            stack.enter_context(mock.patch.object(session.time, "sleep", side_effect=sleep))
            stack.enter_context(mock.patch.object(session.signal, "signal", side_effect=install_signal))
            killed = stack.enter_context(mock.patch.object(session.os, "killpg"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            yield killed

    def state(self):
        return json.loads((self.output / "monitored_state.json").read_text())

    def test_experiment_command_freezes_15k_curriculum_and_resumes_frozen_teacher(self):
        command = session.experiment_command(self.scene, self.sam, self.output)
        expected = {"--semantic_protocol": "v5", "--iterations": "15000",
                    "--semantic_start": "2500", "--semantic_ramp_iterations": "2000",
                    "--validation_views": "12", "--validation_interval": "1000",
                    "--feature_width": "512", "--sam_crop_n_layers": "1",
                    "--checkpoint_interval": "1000"}
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--resume", command)
        self.assertIn("--skip_preprocess", command)
        self.assertNotIn("--init_rgb_ply", command)
        self.assertNotIn("--no_equal_time", command)

    def test_missing_teacher_manifest_prevents_snapshot_and_training(self):
        (self.output / "v5_teacher_files.json").unlink()
        with self.environment(), self.assertRaises(SystemExit):
            session.main()
        self.assertEqual(self.spawned, [])
        self.assertEqual(self.snapshot_calls, 0)

    def test_incomplete_teacher_snapshot_prevents_training_even_with_manifest(self):
        def incomplete(*_args, **_kwargs):
            return SimpleNamespace(returncode=0, stdout=json.dumps({"state": "waiting_for_complete_teacher"}), stderr="")
        with self.environment(snapshot=incomplete), self.assertRaises(RuntimeError):
            session.main()
        self.assertEqual(self.spawned, [])
        self.assertEqual(self.state()["status"], "failed")
        self.assertNotIn("last_snapshot", self.state())

    def test_initial_snapshot_failure_never_claims_completion(self):
        def failed(*_args, **_kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="quota exceeded")
        with self.environment(snapshot=failed), self.assertRaises(RuntimeError):
            session.main()
        state = self.state()
        self.assertEqual(self.spawned, [])
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["computation_complete"])
        self.assertFalse(state["full_final_archive_completed"])
        self.assertIn("quota", state["backup_warning"])

    def test_second_wrapper_cannot_acquire_session_lock(self):
        with (self.output / ".monitored-session.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.environment(), self.assertRaises(SystemExit):
                session.main()
        self.assertEqual(self.spawned, [])

    def test_recorded_live_child_group_blocks_resume(self):
        previous = {"status": "running", "child_pid": 7654321, "git_commit": "abc123"}
        (self.output / "monitored_state.json").write_text(json.dumps(previous))
        with self.environment() as killed, self.assertRaises(SystemExit):
            session.main()
        killed.assert_called_with(7654321, 0)
        self.assertEqual(self.spawned, [])
        self.assertEqual(self.state(), previous)

    def test_changed_code_commit_blocks_resume_before_backup_or_child(self):
        previous = {"status": "failed", "git_commit": "old-code-version", "child_pid": 7654321}
        (self.output / "monitored_state.json").write_text(json.dumps(previous))
        with self.environment() as killed, self.assertRaises(SystemExit):
            killed.side_effect = ProcessLookupError
            session.main()
        self.assertEqual(self.spawned, [])
        self.assertEqual(self.snapshot_calls, 0)
        self.assertEqual(self.state(), previous)

    def test_matching_code_and_dead_old_group_allow_resume(self):
        previous = {"status": "interrupted_check_child_before_resume", "git_commit": "abc123", "child_pid": 7654321}
        (self.output / "monitored_state.json").write_text(json.dumps(previous))
        with self.environment() as killed:
            killed.side_effect = ProcessLookupError
            session.main()
        self.assertEqual(self.state()["status"], "complete")
        self.assertTrue(self.state()["computation_complete"])

    def test_completed_pipeline_keeps_bank_cost_separate_and_does_not_claim_full_archive(self):
        with self.environment():
            session.main()
        state = self.state()
        self.assertEqual(state["status"], "complete")
        self.assertTrue(state["computation_complete"])
        self.assertEqual(self.signal_handlers[signal.SIGTERM], signal.SIG_DFL)
        self.assertEqual(set(state["stages"]), {"benchmark", "descriptor_bank", "descriptor_evaluation", "report"})
        self.assertFalse(state["full_final_archive_completed"])
        self.assertTrue(all(options.get("start_new_session") for _, options in self.spawned))
        main = json.loads((self.output / "comparison.json").read_text())
        self.assertFalse(main["protocol"]["equal_wall_clock"])
        for result in state["stages"].values():
            self.assertIn("observed_wrapper_seconds", result)

    def test_benchmark_exit_without_comparison_stops_before_bank(self):
        self.create_comparison = False
        with self.environment(), self.assertRaisesRegex(RuntimeError, "comparison"):
            session.main()
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(self.state()["status"], "failed")

    def test_nonzero_training_child_prevents_followup_stages(self):
        self.fail_child = "scripts.run_ramen_benchmark"
        with self.environment(), self.assertRaisesRegex(RuntimeError, "benchmark failed"):
            session.main()
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(self.state()["stages"]["benchmark"]["returncode"], 1)
        self.assertNotEqual(self.state()["status"], "complete")

    def test_interrupt_signals_owned_process_group_and_records_resume_guard(self):
        with self.environment(sleep=KeyboardInterrupt) as killed, self.assertRaises(KeyboardInterrupt):
            session.main()
        self.assertIn(mock.call(7654321, signal.SIGTERM), killed.call_args_list)
        self.assertEqual(self.state()["status"], "interrupted_check_child_before_resume")
        self.assertEqual(len(self.spawned), 1)

    def test_sigterm_handler_interrupts_and_stops_the_owned_child_group(self):
        def terminate_during_poll(_seconds):
            handler = self.signal_handlers.get(signal.SIGTERM)
            self.assertTrue(callable(handler), "wrapper must install a SIGTERM-to-interrupt handler")
            handler(signal.SIGTERM, None)

        with self.environment(sleep=terminate_during_poll) as killed, self.assertRaises(KeyboardInterrupt):
            session.main()
        self.assertIn(mock.call(7654321, signal.SIGTERM), killed.call_args_list)
        state = self.state()
        self.assertEqual(state["status"], "interrupted_check_child_before_resume")
        self.assertFalse(state["computation_complete"])
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(self.signal_handlers[signal.SIGTERM], signal.SIG_DFL)

    def test_interrupt_while_recording_new_child_still_terminates_its_group(self):
        original_save = session.save_state
        triggered = False

        def interrupt_after_spawn(path, state):
            nonlocal triggered
            if state.get("status") == "running" and state.get("child_pid") and not triggered:
                triggered = True
                raise KeyboardInterrupt
            return original_save(path, state)

        with self.environment() as killed:
            with mock.patch.object(session, "save_state", side_effect=interrupt_after_spawn):
                with self.assertRaises(KeyboardInterrupt):
                    session.main()
        self.assertTrue(triggered)
        self.assertIn(mock.call(7654321, signal.SIGTERM), killed.call_args_list)
        self.assertEqual(self.state()["status"], "interrupted_check_child_before_resume")
        self.assertFalse(self.state()["computation_complete"])

    def test_uncooperative_child_is_recorded_as_still_running_on_interrupt(self):
        with self.environment(sleep=KeyboardInterrupt) as killed:
            with mock.patch.object(FakeChild, "wait", side_effect=subprocess.TimeoutExpired("fake child", 30)):
                with self.assertRaises(KeyboardInterrupt):
                    session.main()
        state = self.state()
        self.assertTrue(state["child_still_running"])
        self.assertEqual(state["status"], "interrupted_check_child_before_resume")
        self.assertFalse(state["computation_complete"])
        self.assertIn(mock.call(7654321, signal.SIGTERM), killed.call_args_list)

    def test_final_snapshot_error_retains_explicit_backup_warning(self):
        def flaky(*args, **kwargs):
            if self.snapshot_calls:
                return SimpleNamespace(returncode=1, stdout="", stderr="remote quota full")
            return self.complete_snapshot(*args, **kwargs)
        with self.environment(snapshot=flaky):
            session.main()
        state = self.state()
        self.assertEqual(state["status"], "complete_with_backup_warning")
        self.assertTrue(state["computation_complete"])
        self.assertIn("quota", state["backup_warning"])
        self.assertFalse(state["full_final_archive_completed"])
        self.assertIn("last_snapshot", state)  # It is explicitly older, not fabricated.


if __name__ == "__main__":
    unittest.main()
