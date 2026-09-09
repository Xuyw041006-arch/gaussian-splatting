"""Run the frozen Ramen v5 15k experiment with periodic recoverable snapshots.

This runs inside Colab/Linux, not on the user's Mac. It does not keep Colab
alive, bypass runtime limits, delete old results, or purchase resources.
Prepare the full teacher first with run_ramen_benchmark --prepare_only.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]


def experiment_command(scene, sam, output):
    return [sys.executable, "-u", "-m", "scripts.run_ramen_benchmark",
            "--scene", str(scene), "--sam_checkpoint", str(sam), "--output_root", str(output),
            "--semantic_protocol", "v5", "--iterations", "15000",
            "--semantic_start", "2500", "--semantic_ramp_iterations", "2000",
            "--validation_views", "12", "--validation_interval", "1000",
            "--feature_width", "512", "--sam_crop_n_layers", "1",
            "--checkpoint_interval", "1000", "--resume", "--skip_preprocess"]


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(path)


def interrupt_on_sigterm(signum, frame):
    """Convert the notebook's termination into the same owned-child cleanup path."""
    raise KeyboardInterrupt("Monitored session received SIGTERM")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--sam_checkpoint", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--snapshot_dir", required=True, type=Path)
    parser.add_argument("--snapshot_seconds", type=int, default=120)
    args = parser.parse_args()
    if args.snapshot_seconds < 60:
        parser.error("Snapshot cadence must be at least 60 seconds")
    scene, output = args.scene.resolve(), args.output_root.resolve()
    if not (output / "v5_teacher_files.json").is_file():
        parser.error("Complete the isolated full teacher before launching training")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".monitored-session.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This experiment already has an active monitored session; do not start another")
    state_path = output / "monitored_state.json"
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    previous = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("git_commit") != git_commit:
            parser.error("Saved run uses another code commit; restore its exact commit before resuming, or use a new isolated experiment")
        if previous.get("status") not in ("complete", "stage_done") and previous.get("child_pid"):
            try:
                os.killpg(int(previous["child_pid"]), 0)
            except ProcessLookupError:
                pass
            else:
                parser.error("A previously recorded child process group is still alive; inspect it before resuming")
    state = {"status": "starting", "stage": "initial_snapshot",
             "git_commit": git_commit,
             "source_scene": str(scene), "output_root": str(output),
             "snapshot_dir": str(args.snapshot_dir.resolve()),
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "stages": previous.get("stages", {}), "stage_attempts": previous.get("stage_attempts", []),
             "computation_complete": False, "full_final_archive_completed": False}
    snapshot_command = [sys.executable, "-u", "-m", "scripts.snapshot_v5_progress",
                        "--scene", str(scene), "--output_root", str(output),
                        "--destination", str(args.snapshot_dir)]

    def snapshot(required=False):
        try:
            process = subprocess.run(snapshot_command, cwd=REPO, capture_output=True, text=True, timeout=600)
            if process.returncode:
                raise RuntimeError(process.stderr[-3000:] or process.stdout[-3000:])
            payload = json.loads(process.stdout)
            if payload.get("state") != "snapshot_complete":
                raise RuntimeError("Teacher snapshot is not complete: " + payload.get("state", "unknown"))
            state["last_snapshot"] = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "checkpoints": payload["checkpoints"], "teacher": payload["teacher"]}
            state.pop("backup_warning", None)
            print("SNAPSHOT_OK", {name: record.get("iteration", record.get("rgb_export_iteration"))
                                  for name, record in payload["checkpoints"].items()}, flush=True)
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            state["backup_warning"] = str(error)
            print("SNAPSHOT_WARNING: local results retained; remote backup may be older.", str(error), flush=True)
            if required:
                raise
        finally:
            save_state(state_path, state)

    def stage(name, command):
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != git_commit:
            raise RuntimeError("Code commit changed during this run; restore the recorded commit before resuming")
        log = output / ("session_" + name + ".log")
        started = time.monotonic()
        state.update(stage=name, status="running", log=str(log))
        print("STAGE_START", name, command, flush=True)
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                state["child_pid"] = process.pid
                save_state(state_path, state)
                last_snapshot, last_tail = time.monotonic(), ""
                while process.poll() is None:
                    time.sleep(15)
                    with log.open("rb") as recent:
                        recent.seek(max(0, log.stat().st_size - 800))
                        tail = recent.read().decode(errors="replace").replace("\r", "\n")
                    if tail != last_tail:
                        print(tail, flush=True)
                        last_tail = tail
                    if time.monotonic() - last_snapshot >= args.snapshot_seconds:
                        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != git_commit:
                            raise RuntimeError("Code commit changed while training; stopping the owned child group safely")
                        snapshot()
                        last_snapshot = time.monotonic()
            except BaseException:
                # Do not leave an unrecorded training child after notebook interrupt.
                # The benchmark itself launches train.py. Terminate its owned
                # process group, not only the parent and leave a GPU orphan.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    state["child_still_running"] = True
                state["status"] = "interrupted_check_child_before_resume"
                save_state(state_path, state)
                raise
        if name in state["stages"]:
            state["stage_attempts"].append({"stage": name, **state["stages"][name]})
        state["stages"][name] = {"observed_wrapper_seconds": time.monotonic() - started,
                                 "returncode": process.returncode, "log": str(log)}
        state["status"] = "stage_done" if process.returncode == 0 else "failed"
        save_state(state_path, state)
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}; resume only after checking checkpoints")
        snapshot()

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        save_state(state_path, state)
        snapshot(required=True)
        stage("benchmark", experiment_command(scene, args.sam_checkpoint, output))
        if not (output / "comparison.json").is_file():
            raise RuntimeError("Benchmark exited without comparison.json")
        bank = output / "joint/descriptor_bank_15000.npz"
        if not bank.exists():
            stage("descriptor_bank", [sys.executable, "-u", "-m", "scripts.build_semantic_descriptor_bank",
                                      "--model", str(output / "joint"), "--iteration", "15000",
                                      "--output", str(bank)])
        stage("descriptor_evaluation", [sys.executable, "-u", "-m", "scripts.evaluate_lerf_mask",
              "--model", str(output / "joint"), "--iteration", "15000", "--test_mask", str(scene / "test_mask"),
              "--descriptor_bank", str(bank), "--mask_protocol", "gg_native", "--threshold", "0.25",
              "--granularity", "1", "--all_test_rgb", "--output", str(output / "eval_joint_descriptor_bank")])
        stage("report", [sys.executable, "-u", "-m", "scripts.build_ramen_report",
                         "--output_root", str(output), "--report_dir", str(output / "report")])
        state.update(status="complete", stage="complete", computation_complete=True,
                     note="Main comparison and separate bank evaluation finished. Bank cost is additional, not included in baseline equal-time certification. Periodic snapshots are not a complete final-artifact archive.")
        save_state(state_path, state)
        snapshot()
        state["status"] = "complete_with_backup_warning" if state.get("backup_warning") else "complete"
        save_state(state_path, state)
        print("V5_FULL_EXPERIMENT_COMPUTATION_COMPLETE", str(output / "comparison.json"), state["status"], flush=True)
    except BaseException as error:
        if state.get("status") != "interrupted_check_child_before_resume":
            state.update(status="failed", error=str(error))
            save_state(state_path, state)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        lock.close()


if __name__ == "__main__":
    main()
