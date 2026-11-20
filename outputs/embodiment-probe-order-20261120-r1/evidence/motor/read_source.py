"""Read the R2 source-only project volume and write a compact evidence record to stdout."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-dualarm-20261120-r2")
ARMS = ("Panda", "Sawyer")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def rel(path):
    return str(path.relative_to(ROOT))


def read_fit(path):
    manifest = path.parent / "manifest.json"
    meta = load(manifest)["contract"]
    index = load(path)
    fit_id = sha(manifest)
    assert index["fit_id"] == fit_id and index["status"] == "complete"
    rows = []
    for row in index["snapshots"]:
        f = path.parent / row["file"]
        assert f.parent == path.parent / "candidates"
        actual = sha(f)
        assert actual == row["sha256"]
        rows.append([row["step"], actual, f.stat().st_size])
    train_groups = {}
    for arm in ARMS:
        episodes = [r for r in meta["episodes"] if r["body"] == arm and r["split"] == "train"]
        generated = [r for r in episodes if r.get("layout_origin")]
        train_groups[arm] = [len(episodes), len(generated), len({r["seed"] for r in generated})]
    return dict(fit_id=fit_id, index_sha256=sha(path), architecture=meta.get("architecture", "chunk_huber_v1"),
                revision=meta["revision"], snapshots=rows, train_generated_unique=train_groups)


def read_score(path, valid):
    report = load(path)
    stage = path.parent
    phase = stage.parents[2]
    mf = phase / "manifest.json"
    manifest = load(mf)
    rows = manifest["layouts"]
    mode = phase.name
    first, count, threshold = (740000, 50, 38) if mode == "dev" else (840000, 100, 75)
    assert len(rows) == len({r["layout_id"] for r in rows}) == count
    seeds = [r["layout_seed"] for r in rows]
    assert len(set(seeds)) == count and all(first <= s < first + 10000 for s in seeds)
    for key in ("scene_hash", "initial_state_sha256", "movable_xy_hash"):
        assert len({r[key] for r in rows}) == count
    assert set(report["arms"]) == set(ARMS) and report["phase"] == manifest["phase"] == mode
    assert report["status"] == "complete" and report["required_per_arm"] == manifest["required"] == threshold
    assert sha(mf) == load(mf.with_suffix(".sha256.json")) == report["panel_sha256"]
    ck = report["checkpoint_sha256"]
    assert ck == stage.parent.name and (ck, report["step"], report["fit_id"]) in valid
    assert report["stride"] == 5 and stage.name == "stride-5" and report["horizon"] == 300
    contract = load(stage / "contract.json")
    assert all(report[k] == v for k, v in contract.items())
    cd = digest(contract)
    wins = {}
    for arm in ARMS:
        summary = report["arms"][arm]
        episodes = summary["episodes"]
        assert len(episodes) == summary["total"] == count
        by_seed = {r["layout_seed"]: r for r in episodes}
        assert set(by_seed) == set(seeds)
        winners = []
        for r in rows:
            s = r["layout_seed"]
            ep = by_seed[s]
            tape = stage / arm / f"{s}.npz"
            assert ep == load(tape.with_suffix(".json")) and ep["tape"] == str(tape.relative_to(stage))
            assert ep["arm"] == arm and all(ep[k] == r[k] for k in ("layout_id","scene_hash","movable_xy_hash"))
            assert sha(tape) == ep["tape_sha256"]
            with np.load(tape, allow_pickle=False) as z:
                rew, done, suc = (z[k] for k in ("rewards", "done", "success"))
                head = json.loads(z["header"].item())
                physics = head["reset"]["environment_reproducibility"]
                assert head["contract_sha256"] == cd and head["arm"] == arm and head["layout_seed"] == s
                assert head["layout_id"] == r["layout_id"] == physics["init_state_id"]
                assert physics["suite"] == "libero_goal" and physics["task_id"] == 5
                assert physics["perturbation_plan"]["case_id"] == "BASE" and physics["seed"] == s
                assert len(rew) == ep["steps"] and 0 < len(rew) <= 300
                assert head["reset"]["model_identity"]["arm"] == arm
                assert physics["perturbation_plan"]["seed"] == s
                assert all(physics[k] == r[k] for k in ("source_pin", "bddl_sha256", "manifest_sha256"))
                if manifest["revision"] == "e2f794c41c9141a7aadaf67339da919336362f22":
                    assert head["reset"]["admission"] == r["preflight"][arm]
                else:
                    assert np.array_equal(z["native_steps"], np.arange(1, len(rew) + 1))
                assert not done[:-1].any() and done[-1] and not rew[:-1].any() and rew[-1] in (0, 1)
                assert np.array_equal(rew == 1, suc) and bool(suc[-1]) == ep["success"]
                assert rew.sum() == ep["native_return"] and (suc[-1] or len(rew) == 300)
                assert np.array_equal(z["actions"], z["chunks"][:, :5].reshape(-1, 7)[:len(rew)])
            if ep["success"]:
                winners.append(r["layout_id"])
        assert winners == summary["successful_layout_ids"] and len(winners) == summary["successes"]
        wins[arm] = winners
    assert report["passed"] == all(len(ids) >= threshold for ids in wins.values())
    return dict(checkpoint=ck, step=report["step"], passed=report["passed"], phase=mode, stride=5,
                report_sha256=sha(path), panel_sha256=sha(mf), pin=manifest["revision"],
                source50_identity=digest([[r[k] for k in ("layout_seed","layout_id","initial_state_sha256","scene_hash")]
                                          for r in rows]), seed_range=[min(seeds), max(seeds)], total_per_arm=count,
                successful_layout_ids=wins)


def main():
    result = dict(utc=datetime.now(timezone.utc).isoformat(), pfs=str(ROOT),
                  fits={}, campaigns={}, screens={}, panels={}, scores={}, jobs={}, issues={})
    def capture(key, path, fn):
        try:
            result[key][rel(path)] = fn(path)
        except Exception as exc:
            import traceback
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            result["issues"][rel(path)] = [type(exc).__name__, str(exc)[:180], frame.lineno]
    prefixes = ("waypoint","sft","big","host","num","recover","gen","broad","phase","stride","large-panel","wide")
    dirs = [p for p in sorted(ROOT.iterdir()) if p.name.startswith(prefixes)
            and not any(term in p.name.lower() for term in ("ur5","transfer","target","joint")) and p.is_dir()]
    result["root_directory_names"] = sorted(p.name for p in ROOT.iterdir() if p.is_dir()
                                            and not any(x in p.name.lower() for x in ("ur5","transfer","target","joint")))
    for d in dirs:
        launch = d / "launch.txt"
        if launch.exists():
            words = launch.read_text().split()
            assert re.fullmatch(r"[0-9a-f]{40}", words[0]) and re.fullmatch(r"job-[a-z0-9-]+", words[1])
            ex = d / "exit.txt"
            result["jobs"][d.name] = [words[0], words[1], ex.read_text().split() if ex.exists() else None]
        path = d / "fit/candidates.json"
        if path.exists():
            capture("fits", path, read_fit)
        p = d / "banks/campaign.json"
        if p.exists():
            c = load(p)
            stats = []
            for w in c["workers"]:
                bank = p.parent / ("bank%d" % w["worker"])
                idx = bank / "index.json"
                assert sha(idx) == w["index_sha256"]
                index = load(idx)
                assert index["accepted_by_body"] == w["accepted_by_body"]
                stats.append([w["worker"], w["returncode"], w["admitted_layouts"], w["accepted_by_body"],
                              len(w["paired_positive_groups"]), w["index_sha256"]])
            result["campaigns"][rel(p)] = [c["status"], sha(p), stats]
        if d.name.startswith(("phase", "stride")):
            for sub in ("panel", "dev", "native", "check", "screen", "geometry"):
                p = d / sub / "report.json"
                if not p.exists():
                    continue
                r = load(p)
                rec = {k:r[k] for k in ("status", "step", "command", "checkpoint_sha256", "successes",
                                        "starts_per_arm", "stride", "summaries", "paired_success", "positive_coverage") if k in r}
                rec["report_sha256"] = sha(p)
                if "held" in r and isinstance(r["held"], list) and "candidate" in r:
                    rec["checkpoint_sha256"] = r["candidate"]["sha256"]
                result["screens"][rel(p)] = rec
            p = d / "fit/checkpoint.pt"
            if p.exists():
                result["screens"][rel(p)] = {"sha256":sha(p), "bytes":p.stat().st_size}
    valid = {(h, step, f["fit_id"]) for f in result["fits"].values() for step, h, _ in f["snapshots"]}
    for d in dirs:
        if not d.name.startswith("large-panel"):
            continue
        # In the large_panel protocol, test/ is reserved exclusively for Panda/Sawyer source confirmation.
        result["panels"][d.name] = dict(dev=(d/"dev/manifest.json").exists(),
                                       selection=(d/"accepted/selection.json").exists(),
                                       source_confirmation=(d/"test/manifest.json").exists())
        for phase in ("dev", "test"):
            for stage in sorted((d/phase/"scores").glob("*/stride-*")):
                p = stage / "report.json"
                if p.exists():
                    capture("scores", p, lambda path: read_score(path, valid))
                else:
                    result["panels"][d.name].setdefault("incomplete", []).append(
                        [rel(stage), *[len(list((stage/arm).glob("*.npz"))) for arm in ARMS]])
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
