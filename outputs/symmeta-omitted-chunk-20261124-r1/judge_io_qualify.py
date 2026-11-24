"""Exercise the two pinned CLIs on temporary known-answer files, with evaluation labels withheld."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

import judge as j


def inputs(folder):
    rng = np.random.default_rng(8467)
    texture = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    nonce = rng.permutation(1000)
    merged = []
    for role in j.ROLES:
        cell = 14 if role == "fit" else 28
        tasks = np.repeat(j.EXPECTED[role], 2 * cell)
        count = len(tasks)
        arm = np.tile(np.repeat(["Panda", "Sawyer"], cell), len(j.EXPECTED[role]))
        ids = np.array([f"fixture-{role}-{i}" for i in nonce[:count]])
        identity = dict(case_id=ids, physical_trace_id=ids, task=tasks, arm=arm, split=np.repeat(role, count))
        degrees = np.zeros(count)
        gain = np.ones(count)
        base = np.array([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
        cmd = np.broadcast_to(base, (count, 3, 2)).copy()
        before = np.broadcast_to(texture, (count, 3, *texture.shape)).copy()
        after = np.stack([np.roll(texture, (int(y)*2, int(x)*2), axis=(0, 1)) for x, y in base])
        public = dict(**identity, cmd=cmd, valid=np.ones((count, 3), bool), img_before=before,
                      img_after=np.broadcast_to(after, before.shape), dz=np.zeros((count, 3)))
        if role == "fit":
            public.update(true_yaw=degrees, true_gain=gain, world_cmd=cmd)
        else:
            # Cross independent orders of physical cases, labels, seeds and the two reader arrays.
            k = rng.permutation(count)
            label = dict(case_id=ids[k], true_yaw=degrees[k], true_gain=gain[k], true_sign=np.ones(count))
            j.save_arrays(folder / f"labels-{role}.npz", label)
            good = np.arange(count) % 2 == 0
            full = np.stack([-40.*good, 40.*~good], axis=-1).astype(np.float32)
            for seed in j.SEEDS:
                merged.append(dict(**identity, cmd=cmd, valid=public["valid"], seed=np.repeat(seed, count),
                              full_yaw=full, loo_yaw=np.repeat(full[:, None, :], 3, axis=1),
                              full_valid=np.ones((count, 2), bool), loo_valid=np.ones((count, 3, 2), bool),
                              tuned_gain=np.ones(count), tuned_sign=np.ones(count, dtype=np.int32)))
        j.save_arrays(folder / f"{role}.npz", public)
    seal = dict(roles={k: j.EXPECTED[v] for k, v in (("readerfit", "reader_fit"), ("cal", "cal"), ("eval", "eval"))},
                revision="known-answer-fixture", judge_fit_tasks=j.EXPECTED["fit"],
                sha256={p.name: j.digest(p) for p in folder.glob("*.npz")})
    j.write_json(folder / "source-seal.json", seal)
    readers = {k: np.concatenate([m[k] for m in merged]) for k in merged[0]}
    ix = rng.permutation(len(readers["seed"]))
    j.save_arrays(folder / "readers.npz", {k: v[ix] for k, v in readers.items()})
    masks = [dict(seed=s, model=m, native_full_exact=True,
                  folds=[dict(omit=i, poison_exact=True) for i in range(3)])
             for s in j.SEEDS for m in ("tuned", "geom")]
    j.write_json(folder / "prediction-receipt.json",
                 dict(reader_sha256=j.digest(folder / "readers.npz"), revision=seal["revision"],
                      data_seal=j.digest(folder / "source-seal.json"), masks=masks))


def main():
    if not __debug__:
        raise ValueError("Use Python with assertions enabled")
    pin = subprocess.check_output(["git", "-C", j.ROOT, "rev-parse", "HEAD"], text=True).strip()
    runtime = j.execution(pin)
    folder = Path("/mowenjie/fluffy-bassoon-2/outputs") / j.HERE.name / "judge-evidence"
    with TemporaryDirectory(prefix="fixture-", dir=folder) as tmp:
        t = Path(tmp)
        inputs(t)
        # Public sources and receipts persist for both stages. The sealed label path goes away.
        labels = t / "labels-eval.npz"
        held = t / "withheld.npz"
        labels.rename(held)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        def cli(script, *args, check=True):
            result = subprocess.run([sys.executable, "-B", j.HERE / script, "--pin", pin, *args],
                                  capture_output=True, text=True, env=env)
            if check and result.returncode:
                raise ValueError(result.stderr)
            return result
        frozen = t / "frozen"
        cli("judge.py", "--data", t, "--readers", t / "readers.npz", "--out", frozen)
        assert not labels.exists()
        sha = j.digest(frozen / "seal.json")
        # A bad seal must fail before attempting to open the still absent labels.
        rejected = cli("score.py", "--frozen", frozen, "--seal-sha", "0"*64,
                       "--labels", labels, "--out", t / "reject", check=False)
        assert rejected.returncode and "Pre-label seal digest mismatch" in rejected.stderr
        held.rename(labels)
        start = time.monotonic()
        cli("score.py", "--frozen", frozen, "--seal-sha", sha, "--labels", labels, "--out", t / "score")
        r = json.loads((t / "score/metrics.json").read_bytes())
        assert r["joint_refit"]["mae_adaptive"]["ci95"][1] < 1e-10
        assert abs(r["joint_refit"]["mae_tuned"]["mean"] - 20.) < 1e-10
        np.testing.assert_allclose(r["joint_refit"]["hard_concordance"]["mean"], 1., rtol=0., atol=1e-12)
        assert r["registered_advancement_supported"]
        print(json.dumps(dict(runtime=runtime, kind="synthetic file-I/O qualification only",
                   shuffled_reader_rows=2016, evaluation_label_absent_during_freeze=True,
                   incorrect_seal_rejected_before_label_open=True, normal_cli_replay_and_score=True,
                   verified_metrics=["known endpoint MAEs", "hard concordance", "joint intervals", "decision"],
                   bootstrap_draws=r["reps"], score_seconds=round(time.monotonic()-start, 3)), allow_nan=False))


if __name__ == "__main__":
    main()
