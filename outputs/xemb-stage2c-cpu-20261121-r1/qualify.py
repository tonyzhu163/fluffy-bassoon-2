"""Bounded CPU qualification. Each cell has a fresh interpreter and a durable outcome."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "e5500c690c1498eca1b7050350654964190cdf70"
SUITE, TASK, STATE, SEED = "libero_goal", 1, 13, 20500101
CANDIDATES = ("IIWA", "Kinova3")
# Exact BDDL bytes and operator/target identities from the pinned Goal 1 manifest.
CASES = {
    "BASE": ("7a6bb627da6e87025c1cfdd19fff5f8a2c873226734393657505dde2401144a6", "none", []),
    "S13": ("cbd17bd970183214d72775cb6cb450b5a9bfd4af45594896f84d696b720d1d84",
            "asset_or_appearance", ["akita_black_bowl_1", "flat_stove_1"]),
    "S14": ("2b55f3cb69b82bbe2f5dd086d3b339a18a528a8c62aa74f10126868ca3187d04",
            "asset_or_appearance", ["main_table", "akita_black_bowl_1"]),
    "S15": ("f5b6d5c0511cab56f88bc80859c6a0ddc4d78aa23144591744c04636d71b61e7",
            "camera_transform", ["agentview", "robot0_eye_in_hand"]),
    "S16": ("60d8c2ef188659aea3ad2f2d36208a9ca1f40314d7eafb6f3464526c4f8f5e40",
            "camera_transform", ["agentview", "robot0_eye_in_hand"]),
    "S19": ("6d8941b8a0a707c2ce2fab329eb36bcae9551755e3e79b1fd90a0b322c6568e6",
            "observation_fault", ["rgb", "depth"]),
}


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def provenance():
    release = Path(os.environ["CROSSARM_RELEASE"])
    split_file = release / "assets/benchmark_split_v1.json"
    project = json.loads((ROOT / "outputs/libero-meta-learning-20261117-r1/SPLITS.json").read_text())
    official = json.loads(split_file.read_text())["splits"]
    if (git(release, "rev-parse", "HEAD") != PIN
            or hashlib.sha256(split_file.read_bytes()).hexdigest() != project["official_split_sha256"]
            or TASK not in project["development"][SUITE] or TASK not in official["train"][SUITE]
            or TASK in official["test"][SUITE]):
        raise RuntimeError("Development task or pinned split mismatch")
    # The qualified setup imports a hydrated copy of the simulator, separate from Git.
    modules = ["libero/libero/envs/" + name + ".py" for name in
               ("env_wrapper", "bddl_base_domain", "nonphysical_perturbations", "robots/__init__")]
    for name in modules:
        blob = subprocess.check_output(["git", "-C", str(release), "show", PIN + ":simulator-source/" + name])
        if (Path(os.environ["LIBERO_PRO_ROOT"]) / name).read_bytes() != blob:
            raise RuntimeError(f"Simulator source differs from release: {name}")
    files = [*HERE.glob("*.py"), HERE / "run.sh", HERE / "PLAN.md",
             ROOT / "src/fluffy_bassoon/xemb/embodiment.py",
             *(ROOT / "outputs/libero-crossarm-20261120-r1" / name for name in ("arm_env.py", "sim_reset.py"))]
    return dict(code_head=git(ROOT, "rev-parse", "HEAD"), release=PIN,
                file_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})


def main(out):
    out.mkdir(exist_ok=False, parents=True)
    deadline = time.monotonic() + 450
    report = dict(status="running", suite=SUITE, task=TASK, state=STATE, seed=SEED,
                  project_split="development", official_split="train", cases=CASES, rows={})
    summary = out / "results.json"
    save(summary, report)
    rows = report["rows"]

    def cell(arm, case, reason=None):
        key = f"{arm}-{case}"
        remaining = deadline - time.monotonic()
        if reason or remaining < 1:
            rows[key] = dict(status="skipped", reason=reason or "CPU time budget exhausted")
        else:
            path = out / key
            path.mkdir()
            try:
                with (path / "run.log").open("w") as log:
                    rc = subprocess.run([sys.executable, "-Bu", str(HERE / "cell.py"), str(path), arm, case],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode
                result = json.loads((path / "cell.json").read_text()) if (path / "cell.json").exists() else {}
                status = result.get("status", "failed") if rc == 0 else "failed"
                rows[key] = dict(status=status, returncode=rc, evidence=f"{key}/cell.json",
                                 log=f"{key}/run.log", reason=result.get("error", ""))
                if not result:
                    rows[key].update(evidence=None, reason=f"Worker exited {rc} before writing evidence")
            except subprocess.TimeoutExpired:
                rows[key] = dict(status="failed", reason="CPU time budget exhausted", log=f"{key}/run.log")
        print(key, rows[key]["status"], rows[key].get("reason", ""), flush=True)
        save(summary, report)
        return rows[key]["status"] == "passed"

    try:
        report["provenance"] = provenance()
        bases = {arm: cell(arm, "BASE") for arm in ("UR5e", *CANDIDATES)}
        for case in list(CASES)[1:]:
            cell("UR5e", case, None if bases["UR5e"] else "UR5e BASE failed")
        selected = next((arm for arm in CANDIDATES if bases[arm]), None)
        report["selected_new_arm"] = selected
        for arm in CANDIDATES:
            if not bases[arm]:
                cell(arm, "S15", "arm BASE failed")
        if selected:
            for case in ("S15", "S13", "S14", "S16", "S19"):
                cell(selected, case)
            for arm in CANDIDATES:
                if arm != selected and bases[arm]:
                    cell(arm, "S15")
        report["status"] = "passed" if all(r["status"] == "passed" for r in rows.values()) else "incomplete_or_failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(summary, report)
    print(summary, report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
