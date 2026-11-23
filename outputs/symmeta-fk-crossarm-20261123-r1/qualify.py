"""CPU qualification using the historical evaluator checkout for each panel."""
import ast
import inspect
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/mowenjie/fluffy-bassoon-2")
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
PANELS = {
    "gpu09iiwa": ("IIWA", "pi05", "source-wave2d", "490f25c5716aa067199c5b65e32caf6cd15421a4"),
    "gpu06kinova3": ("Kinova3", "pi05", "source-wave2b", "465f33dc657ceab341dbceaf6b98355b94810059"),
    "gpu08pi0fix": ("UR5e", "pi0", "source-wave2d", "490f25c5716aa067199c5b65e32caf6cd15421a4"),
}
CHECKPOINT = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources(panel):
    arm, policy, folder, pin = PANELS[panel]
    archive = ROOT / "outputs" / CAMPAIGN / panel
    src = ROOT / "setups" / CAMPAIGN / folder
    launch = (archive / "launch.txt").read_text().strip()
    config = (archive / "config.txt").read_text().strip()
    require(launch.split()[0] == pin and config.startswith(f"policy={policy} arm={arm} seed=41 ")
            and config.endswith("support=fresh"), (launch, config))
    require(subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip() == pin
            and not subprocess.check_output(["git", "--no-optional-locks", "-C", src, "status", "--porcelain"]),
            f"Dirty or incorrect evaluator checkout {src}")
    sys.path[:0] = [str(src / "src")] + [str(src / "outputs" / name) for name in
        (CAMPAIGN, "libero-crossarm-20261120-r1", "libero-xemb-ur5e-sysid-20261120-r1",
         "libero-xemb-ur5e-scale-20261120-r1")]
    return archive, src, dict(panel=panel, arm=arm, policy=policy, source=str(src), pin=pin,
                              launch=launch, config=config)


def rows(archive, shard):
    base = archive / "live" / f"shard-{shard}"
    hist = load(base / "live-shard.json")
    meta = load(base / "support-meta.json")
    require(hist["shard_idx"] == shard and len(meta) == len(hist["rows"]) == 12, base)
    for i, (m, r) in enumerate(zip(meta, hist["rows"])):
        name = r["embodiment"]
        require(r["index"] == 12 * shard + i and name == m["embodiment"], r["index"])
        for mode in ("active", "passive"):
            path = base / "support" / f"{i:02d}-{name}-{mode}.npz"
            require(Path(m[mode + "_npz"]) == path and path.is_file(), path)
        yield m, r


def historical_functions(ev):
    """Run the original nested loader and predictor directly from the pinned evaluator."""
    body = ast.parse(inspect.getsource(ev.run_shard))
    namespace = vars(ev).copy()
    for name in ("_load_npz", "_pred"):
        node, = (n for n in ast.walk(body) if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, ev.__file__, "exec"), namespace)
    return namespace["_load_npz"], namespace["_pred"]


def reader(panel, output):
    archive, src, info = sources(panel)
    import cv2
    import jax
    import eval_stage2a_live_ur5e as ev

    require(jax.default_backend() == "cpu", "Expected CPU reader")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    checkpoint = archive.parent / "gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack"
    require(sha(checkpoint) == CHECKPOINT, checkpoint)
    params, cfg, _ = ev.load_v2_checkpoint(checkpoint)
    native_load, native_predict = historical_functions(ev)
    tasks, plan = ev.build_test_plan()
    suites = sorted({s for s, _ in tasks})
    result = []
    for shard in range(8):
        for m, r in rows(archive, shard):
            case = plan[r["index"]]
            require(all(case[k] == r[k] for k in ("index", "embodiment", "suite", "task_id")), case)
            p = Path(m["passive_npz"])
            with np.load(p, allow_pickle=False) as z:
                b = {k: z[k][None] for k in z.files}
            require(b["valid"].dtype == bool and np.array_equal(b["valid"], (np.arange(24) < 3)[None]), p)
            b["chunk_steps"] = np.where(b["valid"], ev.CHUNK_SIZE, 0).astype(np.int32)
            b["dis_flow"] = ev.compute_dis_flow_maps(b["img_before"], b["img_after"], b["valid"])
            yaw, gain, grip = ev.predict_embodiment_v2(params, cfg, b)
            yaw, gain, grip = float(yaw[0]), float(gain[0]), int(grip[0])
            original = native_predict(params, cfg, native_load(str(p)))
            require((yaw, gain, grip) == original and np.isfinite(yaw) and np.isfinite(gain)
                    and gain > 0 and grip in (-1, 1), f"Native CPU identity failed: {p}")
            ref = r["arms"]["ours_equiv_v2_passive"]
            dy = (yaw - ref["yaw_hat"] + 180) % 360 - 180
            dg = gain - ref["scale_hat"]
            # Decimal half-unit of archived rounded values plus float32 computation slack.
            match = abs(dy) <= .0051 and abs(dg) <= .000501 and grip == ref["grip_hat"]
            qid = case["query_init_state_id"]
            qseed = 41800000 + 10000 * suites.index(r["suite"]) + 100 * r["task_id"] + qid
            result.append(dict(index=r["index"], suite=r["suite"], task_id=r["task_id"], embodiment=r["embodiment"],
                support_sha256=sha(p), active_sha256=sha(Path(m["active_npz"])), physics=m["physics"],
                support_seed=case["seed"], support_layout=case["support_init_state_id"], query_layout=qid,
                query_seed=qseed, noise_seed=qseed + ev.NOISE_SEED_OFFSET,
                historical=ref, historical_rounding_match=match, native_cpu_identity=True, yaw_hat=yaw, scale_hat=gain, grip_hat=grip,
                yaw_residual_deg=dy, scale_residual=dg))
        print(panel, "reader", shard, flush=True)
    receipt = dict(**info, checkpoint_sha256=CHECKPOINT, checkpoint=str(checkpoint), rows=result,
                   qualified=len(result) == 96 and all(r["native_cpu_identity"] for r in result))
    with output.open("x") as f:
        json.dump(receipt, f, allow_nan=False)
    require(receipt["qualified"], f"CPU identity or completeness failure; inspect {output}")


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("JAX_PLATFORMS") == "cpu",
            "CPU isolation required")
    reader(sys.argv[1], Path(sys.argv[2]))
