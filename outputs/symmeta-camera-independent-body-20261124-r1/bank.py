"""Physical Kinova3 bank using the primary renderer on the original Kinova collector."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/mowenjie/fluffy-bassoon-2")
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
SOURCE = ROOT / "setups" / CAMPAIGN / "source-wave2b"
SOURCE_PIN = "465f33dc657ceab341dbceaf6b98355b94810059"
ARCHIVE = ROOT / "outputs" / CAMPAIGN / "gpu06kinova3"
GEOMETRY = HERE.parent / "symmeta-camera-pose-pairs-20261124-r1"
CASES_SHA = "7a7f704bab9cc9d33ce6c1fbeb7e9722f22ed72553c67cac8303582a22769795"
NAMES = ("nominal", "orbit-", "orbit+")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    path = HERE / "cases.csv"
    assert sha(path) == CASES_SHA
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in ("index", "shard", "slot", "task_id", "support_layout", "support_seed", "angle_deg",
                  "query_layout", "query_seed", "noise_seed"):
            row[k] = int(row[k])
    assert len(rows) == 64 and len({r["index"] for r in rows}) == 64
    for shard in range(8):
        for slots in (range(4, 8), range(8, 12)):
            sub = [r for r in rows if r["shard"] == shard and r["slot"] in slots]
            assert len(sub) == 4 and sorted(r["angle_deg"] for r in sub) == [-36, -36, 36, 36]
    return rows


def collect(out):
    assert os.sched_getaffinity(0) <= {36, 37} and os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and not sys.flags.optimize
    assert subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True).strip() == SOURCE_PIN
    assert not subprocess.check_output(["git", "--no-optional-locks", "-C", SOURCE, "status", "--porcelain"])
    assert (ARCHIVE / "launch.txt").read_text().split()[0] == SOURCE_PIN
    assert (ARCHIVE / "config.txt").read_text().startswith("policy=pi05 arm=Kinova3 seed=41 ")
    sys.path[:0] = [str(GEOMETRY), str(SOURCE / "outputs/libero-crossarm-20261120-r1"),
                    str(SOURCE / "src/fluffy_bassoon/xemb")]
    import render
    from geometry import as_h, dump, project
    from world_control import panel
    from arm_env import ArmEnv
    from OpenGL import GL

    class KinovaViews(render.Views):
        def __init__(self, arm, first, source):
            super().__init__(arm, first, source)
            self.orbit = 36
            body = first["model_identity"]
            assert body["arm"] == "Kinova3" and body["joint_count"] == 7
            gl = GL.glGetString(GL.GL_RENDERER).decode()
            assert "llvmpipe" in gl.lower()
            self.reset_info = dict(renderer=gl, body=body, q0=first["observation"]["robot0_joint_pos"].tolist(),
                                   reset=first["environment_reproducibility"])

    class CapturedViews(KinovaViews):
        def __init__(self, arm, first, source):
            super().__init__(arm, first, source)
            reset_info.update(self.reset_info)

    # Limit the upstream collector's process-local camera enumeration. Its underlying
    # sensor hook, MuJoCo state check and independent landmark renderer are unchanged.
    expected = cases()
    archive_entries = {}
    for c, entries in panel(SOURCE, ARCHIVE / "live", range(8)):
        for e in entries[4:]:
            archive_entries[e["index"]] = c, e
    rows, diags = [], []
    reset_info = {}
    with patch.object(render, "VIEWS", NAMES), patch.object(render, "Views", CapturedViews):
        for case in sorted(expected, key=lambda r: (r["slot"] < 8, r["shard"], r["slot"])):
            i = case["index"]
            c, e = archive_entries[i]
            assert (c["suite"], c["task_id"], c["seed"], c["init_state_id"], e["embodiment"], e["physics"]) == (
                case["suite"], case["task_id"], case["support_seed"], 11, case["embodiment"], case["physics"])
            assert sha(e["passive"]) == case["support_sha256"]
            reset_info.clear()
            row, diag = render.collect(out, "Kinova3", c, e, None)
            assert row["archive_sha256"] == case["support_sha256"] and row["physics"] == reset_info["reset"]["initial_state_fingerprint"]
            row.update(role="heldout", source_collector_pin=SOURCE_PIN, assigned_view="orbit+" if case["angle_deg"]>0 else "orbit-",
                       query_layout=case["query_layout"], query_seed=case["query_seed"], noise_seed=case["noise_seed"],
                       regime=case["regime"], support_manifest_sha256=sha(Path(e["passive"]).parents[1]/"support-meta.json"))
            diag.update(reset_info)
            assert set(row["views"]) == set(NAMES) and [m["view"] for m in diag["landmarks"]] == list(NAMES)
            t = np.asarray(row["frame_times_s"])
            assert len(t) == 4 and np.all(np.diff(t)>0)
            xyz = np.asarray(diag["world_tcp_m"])
            views = row["views"]
            ref = project(views["nominal"]["P"], xyz)
            diag["tcp_error_px"] = {}
            with np.load(views["nominal"]["passive_npz"]) as nom, np.load(e["passive"]) as src:
                assert np.array_equal(nom["issued_mean"], src["issued_mean"][:3])
                np.testing.assert_array_equal(diag["issued_commands"], np.repeat(src["issued_mean"][:3], 5, axis=0))
                assert np.asarray(diag["physical_commands"]).shape == (15, 7)
                for name, v in views.items():
                    assert sha(v["passive_npz"]) == v["sha256"]
                    with np.load(v["passive_npz"]) as moved:
                        assert moved["joint_count"] == 7 and np.all(moved["chunk_steps"] == 5)
                        assert moved["valid"].all() and moved["delta_joint"].shape == (3, 7)
                        for key in ("issued_mean","valid","delta_eef","delta_jaw","delta_joint","frame0_state","chunk_steps"):
                            np.testing.assert_array_equal(moved[key], nom[key])
                        np.testing.assert_array_equal(moved["frame0_img"], moved["img_before"][0])
                        np.testing.assert_array_equal(moved["img_before"][1:], moved["img_after"][:-1])
                    obs = project(v["P"], xyz)
                    assert np.all((obs >= 0) & (obs <= 63))
                    opp = {"nominal":"nominal","orbit-":"orbit+","orbit+":"orbit-"}[name]
                    hs = {"correct":v["H_view_to_ref"], "opposite":views[opp]["H_view_to_ref"], "identity":np.eye(3)}
                    diag["tcp_error_px"][name] = {k:np.linalg.norm(project(as_h(np.asarray(h)), obs)-ref,axis=1).tolist()
                                                  for k,h in hs.items()}
            rows.append(row)
            diags.append(diag)
            # A separate immutable manifest for each complete, prespecified stratum.
            dump(out/f"{row['id']}/diagnostics.json", diag)
            dump(out/f"{row['id']}/row.json", row)
            dump(out/"progress.json", dict(count=len(rows), last=i))
            if len(rows) in (1,32,64):
                digest_by_id = {r["id"]:sha(out/r["id"]/"diagnostics.json") for r in rows}
                dump(out/f"bank{len(rows)}.json", dict(rows=rows, diagnostic_sha256=digest_by_id, cases_sha256=CASES_SHA))
            print("qualified", row["id"], len(rows), flush=True)
    # Validate released query layouts and body identity without stepping a query.
    resets = []
    for shard in range(8):
        for layout in (26,27):
            case = next(r for r in expected if r["shard"] == shard and r["query_layout"] == layout)
            with ArmEnv("Kinova3", case["suite"], case["task_id"]) as arm:
                first = arm.reset(layout, case["query_seed"])
                rr = first["environment_reproducibility"]
                assert rr["seed"] == case["query_seed"] and rr["init_state_id"] == layout
                assert first["worker_step_index"] == 0 and first["model_identity"]["joint_count"] == 7
                resets.append(dict(shard=shard, reset=rr, model=first["model_identity"], horizon=first["max_steps"],
                                   prompt=first["language_instruction"]))
    dump(out/"query-resets.json", resets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    collect(parser.parse_args().out)
