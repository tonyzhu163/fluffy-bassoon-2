#!/usr/bin/env python
"""Run an arm and its reward worker inside one finite scheduler allocation.

S uses one GPU. F uses policy GPU0 and a persistent official scorer on GPU1,
in their separate qualified Python environments. The filesystem carries only
complete pending batches/labels; no per-batch allocation or manual handoff.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import paired_stopping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    source = Path(__file__).resolve().parents[2]
    arm, exchange = profile["arm"], Path(profile["exchange"])
    if arm not in ("S", "F") or profile["gpus"] != (2 if arm == "F" else 1):
        raise ValueError("One policy GPU per arm, plus a dedicated F scorer GPU required")
    exchange.mkdir(parents=True, exist_ok=True)
    deadline = profile["deadline_unix"] - profile["closure_seconds"]
    children = []

    def start(python, script, child_profile, env):
        path = exchange / f"{arm}-{script}-{len(children)}.json"
        write_new(path, child_profile)
        child = subprocess.Popen([python, "-B", str(source / "scripts/diagnostics" / (script + ".py")),
                                  "--profile", str(path)], env={**os.environ, **env}, start_new_session=True)
        children.append(child)
        return child

    def wait_file(path, watched=()):
        while time.time() < deadline:
            if path.exists():
                return json.loads(path.read_text())
            for child in watched:
                if child.poll() is not None:
                    raise RuntimeError(f"Child exited {child.returncode} before {path}")
            peer = exchange / ("S-terminal.json" if arm == "F" else "F-terminal.json")
            if peer.exists():
                raise RuntimeError(f"Peer pipeline closed before {path}; no idle continuation")
            time.sleep(1)
        raise TimeoutError(f"Finite allocation deadline before {path}")

    def child_result(child):
        while child.poll() is None and time.time() < deadline:
            if scorer is not None and scorer.poll() not in (None, 0):
                raise RuntimeError("Persistent reward worker failed; keep the complete pending batch, no fallback")
            if arm == "S":
                coordinate()
            time.sleep(1)
        if child.poll() is None:
            raise TimeoutError("Driver missed its closed-boundary reserve")
        if child.returncode:
            raise RuntimeError(f"Driver exited {child.returncode}; recovery artifacts retained")

    def coordinate():
        common = Path(profile["common_before"]["output"])
        initial = common / "initial_DEV-000/evaluation.json"
        if not initial.exists():
            return
        rows = {key: [json.loads(initial.read_text())] for key in ("S", "F")}
        stop_root = Path(profile["training"]["paired_stop_root"])
        for iteration in (2, 4, 6, 8, 10):
            target = stop_root / f"{iteration:03d}.json"
            panels = {key: Path(profile["arm_outputs"][key]) / f"DEV-{iteration:03d}/evaluation.json"
                      for key in ("S", "F")}
            if all(path.exists() for path in panels.values()):
                for key in rows:
                    rows[key].append(json.loads(panels[key].read_text()))
                if not target.exists():
                    write_new(target, paired_stopping(rows["S"], rows["F"]))
            elif any((Path(profile["arm_outputs"][key]) / "FINAL.json").exists()
                     or (exchange / f"{key}-terminal.json").exists() for key in ("S", "F")):
                if not target.exists():
                    write_new(target, dict(stop=True, reason="peer_closed_or_finite_clock",
                        interpretation="Resource censoring, not scientific convergence"))
                break
            else:
                break

    gate_path = Path(profile["scorer"]["output"]) / "baseline-validation.json"
    scorer = None
    terminal = dict(completed=False, arm=arm)
    try:
        if arm == "F":
            # Validate the newly added engine-reuse core on allocated Linux,
            # without loading another model or rerunning earlier suites.
            carried = profile.get("persistent_core_qualification")
            if carried:
                import xml.etree.ElementTree as ET
                if sha(carried["path"]) != carried["sha256"]:
                    raise ValueError("Carried engine core qualification changed")
                suite = ET.parse(carried["path"]).getroot().find("testsuite")
                if suite is None or any(int(suite.attrib[key]) != expected
                        for key, expected in dict(tests=2, errors=0, failures=0, skipped=0).items()):
                    raise ValueError("Exactly the already-passed two affected core cases required")
            else:
                subprocess.run([profile["policy_python"], "-B", "-m", "pytest", "-q",
                    str(source / "tests/test_flora_terminal_worker.py"),
                    "--junitxml=" + str(exchange / "persistent-worker-core.xml")],
                    env={**os.environ, **profile["policy_env"]}, check=True,
                    timeout=min(120, max(1, deadline - time.time())))
            worker_profile = dict(profile["scorer"], mode="integrated")
            # These arguments belong only to calibration. The F policy profile
            # and training label requests contain no calibration truth path.
            calibration_args = ["--episodes", profile["native_episodes"],
                                "--baseline-monitor", profile["native_monitor"]]
            worker_path = exchange / "F-scorer-profile.json"
            write_new(worker_path, worker_profile)
            scorer = subprocess.Popen([profile["scorer_python"], "-B",
                str(source / "scripts/diagnostics/robocasa_residual_score.py"),
                "--profile", str(worker_path), *calibration_args],
                env={**os.environ, **profile["scorer_env"]}, start_new_session=True)
            children.append(scorer)
        gate = wait_file(gate_path, [scorer] if scorer else [])
        if gate.get("go") is not True:
            terminal.update(completed=True, stop_reason="native8_Flora_AUROC_gate_failed", gate=gate)
            return
        gate_ref = dict(path=str(gate_path), sha256=sha(gate_path))
        common_done = exchange / "common-before-complete.json"
        if arm == "S":
            common = start(profile["policy_python"], "robocasa_residual_ppo",
                           dict(profile["common_before"], baseline_validation=gate_ref), profile["policy_env"])
            child_result(common)
            write_new(common_done, dict(completed=True, output=profile["common_before"]["output"]))
        else:
            wait_file(common_done, [scorer])
        policy = start(profile["policy_python"], "robocasa_residual_ppo",
                       dict(profile["training"], baseline_validation=gate_ref), profile["policy_env"])
        child_result(policy)
        final = Path(profile["training"]["output"]) / "FINAL.json"
        if not final.exists():
            raise RuntimeError("A successful arm must retain its final policy/critic/Adam/RNG checkpoint")
        terminal.update(completed=True, final=json.loads(final.read_text()))
    except BaseException as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_new(exchange / f"{arm}-terminal.json", terminal)
        if scorer is not None and scorer.poll() is None:
            write_new(Path(profile["scorer"]["output"]) / "shutdown.json", dict(reason="F policy closed"))
            try:
                scorer.wait(timeout=max(1, min(120, deadline - time.time())))
            except subprocess.TimeoutExpired:
                # Process groups are our own children. The scheduler wrapper
                # owns final TERM/KILL cleanup and the inclusive GPU clock.
                scorer.terminate()
        for child in children:
            if child.poll() is None:
                child.terminate()


if __name__ == "__main__":
    main()
