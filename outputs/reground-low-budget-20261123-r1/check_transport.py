"""CPU qualification with real weights and a one-step scene/policy test double."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import runner
import torch
from common import SOURCE_STORE, WORK, configure, ref


class Scene:
    def __init__(self):
        self.calls = 0

    def call(self, idx, cmd, args):
        x = np.zeros(25, np.float32)
        x[3] = x[7] = 1
        if cmd == "reset":
            self.calls += 1
            result = {"x": x, "obs": {}, "description": "CPU fixture", "max_steps": 1}
        elif cmd == "step":
            result = {"x": x, "success": True}
        else:
            ref.require(cmd == "summary", "Unexpected full simulator state access")
            result = {
                "speeds": np.array([0.3], np.float32),
                "peak_speed": 0.3,
                "displaced": 0,
                "fallen": 0,
                "contact_steps": 0,
                "table_steps": 0,
            }
        return [dict(result) for _ in idx]


def check():
    from reground.data import babble
    from reground.eval import jobs
    from reground.method import build
    from reground.models import cache

    inputs = runner.Inputs(0, 15)
    ref.require(
        not torch.cuda.is_available() and len(os.sched_getaffinity(0)) <= 2,
        "Two CPU cores only",
    )
    with TemporaryDirectory(dir=WORK) as tmp:
        d = Path(tmp)
        # Exercise a fresh complete fit on the shipping input path and compare state
        # tensors to a literal release fit with its original raw pools and holdout.
        literal = WORK / "literal-seed0-b15"
        native_equal = {}
        for k in ("fwd", "idm"):
            runner.fit_one(inputs, d, k, "cpu")
            key = inputs.record["windows"][k]["spec_hash"]
            a = literal / "models" / key
            b = d / "models" / key

            def flatten(x, prefix=""):
                for name, v in x.items():
                    field = prefix + name
                    if isinstance(v, dict):
                        yield from flatten(v, field + "/")
                    elif torch.is_tensor(v):
                        yield field, v

            av = dict(
                flatten(
                    torch.load(a / "model.pt", weights_only=False, map_location="cpu")
                )
            )
            bv = dict(
                flatten(
                    torch.load(b / "model.pt", weights_only=False, map_location="cpu")
                )
            )
            ref.require(
                av.keys() == bv.keys() and all(torch.equal(av[p], bv[p]) for p in av),
                "Weight mismatch",
            )
            l, small = (
                ref.read(a / "fit_metrics.json"),
                ref.read(b / "fit_metrics.json"),
            )
            ref.require(
                "onpolicy" in l
                and "onpolicy" not in small
                and all(
                    l[key] == small[key]
                    for key in ("fit", "heldout", "n_train_eps", "n_val_eps")
                ),
                "Metric mismatch",
            )
            native_equal[k] = {
                "tensors": len(av),
                "epochs": l["fit"]["epochs"],
                "literal_onpolicy": l["onpolicy"],
            }
        with runner.bounded(inputs, fit=False):
            for f in (
                inputs.tape,
                SOURCE_STORE / "runs/data/rot_z/90/seed0/babble/shard000.npz",
            ):
                try:
                    with f.open("rb"):
                        raise AssertionError(f"Opened calibration: {f}")
                except RuntimeError as exc:
                    ref.require("Unexpected calibration" in str(exc), exc)
            for fn in (
                build.estimated_delay,
                build.adaptation_data,
                babble.load_pool,
                cache.heldout_vla_pool,
            ):
                try:
                    fn()
                    raise AssertionError("Calibration loader accepted")
                except RuntimeError as exc:
                    ref.require("Unbudgeted" in str(exc), exc)
        panel = runner.panel_jobs(inputs, "panel32")
        full = runner.panel_jobs(inputs, "full")
        ref.require(
            len(panel) == 8
            and sum(len(j["episodes"]) for j in full) == 280
            and all(j["episodes"] == full[j["shard"]]["episodes"][:4] for j in panel),
            "Grid mismatch",
        )
        scene = Scene()
        ctx = jobs.WorkerCtx("pi05", "cpu", 4, print)
        ctx._pool = scene
        ctx._vla = SimpleNamespace(
            revision=ref.POLICY,
            predict_chunk=lambda obs, *_: np.full((len(obs), 50, 7), 0.05, np.float32),
        )
        with patch.object(
            runner,
            "panel_jobs",
            side_effect=lambda i, p: [panel[0] if p == "panel32" else full[0]],
        ):
            store = runner.evaluate(inputs, d, "cpu", "panel32", ctx)
            ref.require(scene.calls == 1, "Expected one real fixture batch")
            runner.evaluate(inputs, d, "cpu", "full", ctx)
            ref.require(scene.calls == 4, "Panel batch was not reused")
            runner.evaluate(inputs, d, "cpu", "full", ctx)
            ref.require(scene.calls == 4, "Complete resumption reran a batch")
        with (
            runner.deployment(inputs, d, "cpu"),
            patch.object(jobs, "job_dir", lambda _: d / "direct"),
        ):
            jobs.run_eval_job(full[0], ctx)
        saved, direct = (
            jobs.load_traces(p) for p in [store / "full/shard000", d / "direct"]
        )
        ref.require(
            len(saved) == 16 and direct.keys() == saved.keys(), "Incomplete fixture"
        )
        for k, t in direct.items():
            for key in t:
                np.testing.assert_array_equal(t[key], saved[k][key])
        return {
            "literal_vs_bounded": native_equal,
            "negative_access_tests": 6,
            "prefix_reuse_new_batches": 3,
            "full_repeat_new_batches": 0,
            "compared_episodes": 16,
            "exact_trace_fields": list(next(iter(saved.values()))),
            "panel_shards": [j["shard"] for j in panel],
            "full_grid": 280,
            "benchmark_episodes": 0,
        }


if __name__ == "__main__":
    configure()
    ref.write(WORK / "transport-cpu.json", check())
