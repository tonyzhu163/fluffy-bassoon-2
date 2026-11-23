"""Baige CPU only. Compare the two original evaluators and the native CEM with/without replay."""
import csv
from hashlib import sha256
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import numpy as np
import reference as ref


class Scene:
    """Deterministic, action-dependent 21-control episodes, including two live re-queries."""
    def obs(self, i):
        im = np.roll(self.images[i], self.steps[i], axis=1)
        return dict(pixels=dict(image=im, image2=im[::-1].copy()), robot_state=dict(
            eef=dict(pos=self.x[i, :3].astype(np.float64)),
            gripper=dict(qpos=self.x[i, 9:11].astype(np.float64))))

    def call(self, idx, cmd, args):
        if cmd == "reset":
            self.tasks = [a[0] for a in args]
            self.steps = [0] * len(idx)
            self.x = np.zeros((len(idx), 25), np.float32)
            self.x[:, [3, 7]] = 1
            self.x[:, 0] = [a[2] / 100 for a in args]
            self.images = [np.random.default_rng(a[2]).integers(0, 256, (256, 256, 3), np.uint8) for a in args]
            return [dict(x=self.x[i].copy(), obs=self.obs(i), description=self.tasks[i], max_steps=21) for i in idx]
        if cmd == "full_state":
            return [dict(task=self.tasks[i], state=dict(sim=self.x[i].astype(np.float64), shift=None)) for i in idx]
        if cmd == "summary":
            return [dict(speeds=np.zeros(self.steps[i], np.float32), peak_speed=0., displaced=0, fallen=0,
                         contact_steps=0, table_steps=0) for i in idx]
        assert cmd == "step"
        result = []
        for i, (action, render) in zip(idx, args, strict=True):
            self.steps[i] += 1
            self.x[i, :3] += np.array([-action[1], action[0], action[2]]) * .02
            out = dict(x=self.x[i].copy(), success=self.steps[i] == 21)
            if render:
                out["obs"] = self.obs(i)
            result.append(out)
        return result


def fixture_cache(cache):
    import torch
    from reground.models.dynamics import ForwardEnsemble, ModelCfg
    from reground.models.features import DX_DIM, Normaliser
    from reground.models.idm import IDM

    torch.manual_seed(41)
    models = []
    for role, cell, budget in (("nom", "none/0", 100000), ("dep", "rot_z/90", 10000)):
        for kind in ("fwd", "idm"):
            spec = cache.make_spec(kind, role, cell, 0, budget, .6)
            cfg = ModelCfg(**spec["cfg"])
            model = (ForwardEnsemble if kind == "fwd" else IDM)(cfg)
            model.in_norm = Normaliser(torch.zeros(model.in_dim), torch.ones(model.in_dim))
            if kind == "fwd":
                model.out_norm = Normaliser(torch.zeros(DX_DIM), torch.full((DX_DIM,), .01))
            path = cache.spec_dir(spec)
            ref.write(path / "spec.json", spec)
            torch.save(model.state_dict(), path / "model.pt")
            models.append(dict(dir=str(path.relative_to(ref.STORE)), kind=kind, role=role))
    ref.write(ref.output(dict(kind="fit", seed=0)), dict(models=models))


def check(tmp):
    import torch
    from reground.eval import jobs
    from reground.method import build
    from reground.models import cache
    from reground.models.dynamics import ModelCfg
    from reground.policies.vla import VLA
    from reground.utils import noise_seed
    import symmeta_adapter as stream
    import runner
    from opening import Opening, identical

    class Policy:
        revision, device, chunk_size, noise_dim = "cpu-fixture", "cpu", 50, 32
        _noise = VLA._noise

        def predict_chunk(self, obs, tasks, seeds, q):
            self.calls.append(q)
            actions = self._noise(seeds, q).numpy()[:, :, :7].copy() * .15
            actions[:, :, 0] += .25
            for i, o in enumerate(obs):
                actions[i, :, :3] += o["robot_state"]["eef"]["pos"]
            return actions

    def context():
        ctx = jobs.WorkerCtx("pi05", "cpu", 4, lambda m: None)
        ctx._pool, ctx._vla = Scene(), Policy()
        ctx.vla.calls = []
        return ctx

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and not torch.cuda.is_available()
    torch.set_num_threads(1)
    base = ref.units("base", [0])[4]
    sym = stream.units([0])[4]
    rg = ref.units("reground", [0])[4]
    # Confirm selection against the two immutable original shards before switching to private models.
    filters = {}
    for seed, shard, expected in ((0, 4, [("10_3", i, 0) for i in range(44, 50)]),
                                  (1, 15, [("spatial_0", 48, 0)])):
        job = ref.units("base", [seed])[shard]
        _, affected = runner.original_inputs(job)
        assert set(expected) <= set(affected) and len(job["episodes"]) == 16
        filters[f"{seed}/{shard}"] = affected

    original = {}
    for j in range(0, 16, 4):
        keys = list(map(tuple, base["episodes"][j:j+4]))
        scene = Scene()
        outs = scene.call(range(4), "reset", [(t, base["cell"], i, 0, True) for t, i, r in keys])
        for key, out in zip(keys, outs):
            obs = out["obs"]
            original[key] = dict(x=out["x"], nonce=noise_seed(*key[1:]),
                camera=sha256(obs["pixels"]["image"].tobytes()).hexdigest(),
                eef=obs["robot_state"]["eef"]["pos"], jaw=obs["robot_state"]["gripper"]["qpos"])

    store = tmp / "shards"
    store.mkdir()
    ctx = context()
    runner.stage("capture", ctx, base, original, tmp / "record", store)
    assert ctx.vla.calls == [0] * 4
    comparisons = {}
    with stream.reader_process() as reader, patch.object(ref, "STORE", tmp / "cache"), \
            patch.object(cache, "RUNS", tmp / "cache/runs"), \
            patch.object(cache, "default_cfg", lambda *args: ModelCfg(width=8, depth=1, ens=1, L=1)):
        for arm, job in (("base", base), ("symmeta", sym), ("reground", rg)):
            if arm == "reground":
                fixture_cache(cache)  # only now do the four private fake weight files exist
            direct = tmp / ("stock-" + arm)
            raw = context()
            build._MEMO.clear()
            if arm == "symmeta":
                stream.run_job(job, raw, *reader, direct, fixture=True)
            else:
                with patch.object(jobs, "job_dir", lambda _: direct):
                    jobs.run_eval_job(job, raw)
            paired = context()
            runner.stage(arm, paired, job, original, tmp / ("work-" + arm), store, reader, fixture=True)
            assert raw.vla.calls == [0, 1, 2] * 4 and paired.vla.calls == [1, 2] * 4
            left, right = jobs.load_traces(direct), jobs.load_traces(store / arm)
            assert left.keys() == right.keys() and len(right) == 16
            for key in left:
                assert all(identical(left[key][f], right[key][f]) for f in ("X", "A", "A_vla", "ref", "speed"))
            def rows(path):
                with (path / "results.csv").open() as f:
                    return [{k: v for k, v in row.items() if k not in ("ms_per_step", "ms_vla", "wall_s")}
                            for row in csv.DictReader(f)]
            assert rows(direct) == rows(store / arm)
            comparisons[arm] = len(right)
            if arm == "reground":
                assert any(not identical(t["A"][:10], t["A_vla"][:10]) for t in right.values())
            if arm == "symmeta":
                costs = ref.read(store / arm / "calibration.json")
                assert all([u["end_control"] for u in c["updates"]] == [5, 10, 15, 20] for c in costs)
                assert any(c["first_changed_control"] is not None for c in costs)

    rejected = []
    for fault in ("camera", "wrist", "jaw", "eef", "sim", "task", "nonce", "noise"):
        ctx = context()
        work = tmp / ("wrong-" + fault)
        work.mkdir()
        gate = Opening(ctx, base, original, work, store / "capture")
        args = [(t, base["cell"], i, 0, True) for t, i, r in base["episodes"][:4]]
        original_call = gate.call

        def faulty_call(idx, cmd, args):
            outs = original_call(idx, cmd, args)
            if cmd == "reset":
                o = outs[0]["obs"]
                if fault in ("camera", "wrist"):
                    o["pixels"]["image" if fault == "camera" else "image2"][0, 0, 0] ^= 1
                if fault == "eef":
                    o["robot_state"]["eef"]["pos"][0] += .01
                if fault == "jaw":
                    o["robot_state"]["gripper"]["qpos"][0] += .01
                if fault == "task":
                    outs[0]["description"] += " altered"
            if cmd == "full_state" and fault == "sim":
                outs[0]["state"]["sim"][1] = 1
            return outs

        gate.call = faulty_call
        if fault == "noise":
            ctx.vla._noise = lambda seeds, q: VLA._noise(ctx.vla, seeds, q) + 1
        try:
            outs = gate.reset(range(4), "reset", args)
            seeds = [noise_seed(i, r) for _, i, r in base["episodes"][:4]]
            seeds[0] += int(fault == "nonce")
            gate.query([o["obs"] for o in outs], [o["description"] for o in outs], seeds, 0)
        except ValueError as exc:
            assert "Opening differs at batch" in str(exc) and (work / "mismatch-0.npz").is_file()
            rejected.append(fault)
        else:
            raise AssertionError(f"Accepted altered {fault}")
    return dict(host=socket.gethostname(), code=os.environ["SYMMETA_CODE_REV"], upstream=ref.PIN,
                reader=reader[1], gpu_hours=0, original_filter=filters, byte_exact_native_shard_rows=comparisons,
                fixture="deterministic scene and policy, archive reader, native CEM with private synthetic cache",
                real_queries_per_batch=dict(capture=[0], stock=[0, 1, 2], controlled=[1, 2]),
                rejected=rejected)


if __name__ == "__main__":
    ref.configure()
    tmp = Path(sys.argv[1])
    tmp.mkdir()
    ref.write(tmp / "passed.json", check(tmp))
    print(tmp / "passed.json")
