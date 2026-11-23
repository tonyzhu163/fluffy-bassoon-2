"""Baige CPU qualification. True angles occur only in independent inverse test fixtures."""
from contextlib import contextmanager
import os
import socket
import sys
import time

import numpy as np


def reciprocal():
    from reground.envs.libero_env import TaskEnv
    from reground.envs.shifts import RotZ
    from symmeta_adapter import policy_observation
    from retry import issue

    rng = np.random.default_rng(417)
    proposal = rng.uniform(-.45,.45,(64,7)).astype(np.float32)
    records = {}
    for yaw in (-137., -90., 0., 37., 90., 168.):
        base = issue(proposal, (yaw,1,1), "translation")
        wrist = issue(proposal, (yaw,1,1), "translation_wrist")
        native = np.stack([RotZ(yaw).transform(a) for a in wrist])
        error = np.max(abs(native-proposal))
        # One float32 matrix multiply; tolerance scales with the machine precision.
        np.testing.assert_allclose(native, proposal, rtol=0, atol=4*np.finfo(np.float32).eps)
        np.testing.assert_array_equal(base[:,3:6],proposal[:,3:6])
        np.testing.assert_array_equal(base[:,[0,1,2,5,6]],wrist[:,[0,1,2,5,6]])
        records[str(yaw)] = float(error)
    saturation = rng.uniform(-2,2,(64,7)).astype(np.float32)
    for gain, sign in ((.5,-1),(2,1)):
        a,b = [issue(saturation, (37,gain,sign), arm) for arm in ("translation","translation_wrist")]
        np.testing.assert_array_equal(a[:,[0,1,2,5,6]],b[:,[0,1,2,5,6]])
        np.testing.assert_array_equal(b[:,6],np.clip(saturation[:,6],-1,1)*sign)
        assert np.max(abs(b)) <= 1

    # Independently instantiate the same released goal layout for each trajectory.
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
    from opening import observations, identical
    tape = np.array([[.12,-.07,.04,.16,-.12,.09,-1.]] * 10, np.float32)
    states, speeds = {}, {}
    for mode, cell in (("nominal","none/0"),("repeat","none/0"),("translation","rot_z/90"),
                       ("translation_wrist","rot_z/90")):
        env = TaskEnv("goal_2",cell)
        try:
            o = env.reset(15,0)
            gl = [glGetString(k).decode() for k in (GL_VENDOR,GL_RENDERER)]
            assert "llvmpipe" in gl[1].lower()
            xs = [o["x"]]
            anchor = o["obs"]["robot_state"]["eef"]["pos"].copy()
            cmds = tape if cell == "none/0" else issue(tape,(90.,1.,1),mode)
            for i, a in enumerate(cmds):
                o = env.step(a, render=i+1==len(tape))
                xs.append(o["x"])
            states[mode] = np.stack(xs)
            speeds[mode] = env.summary()["speeds"]
            raw = o["obs"]
            mapped = policy_observation(raw,anchor,90.)
            r,m = observations([raw]),observations([mapped])
            pos = "0/obs/robot_state/eef/pos"
            assert r.keys()==m.keys() and all(identical(r[k],m[k]) for k in r if k != pos)
            delta = r[pos].astype(np.float32)-anchor.astype(np.float32)
            expected = anchor.astype(np.float32)+np.array([-delta[1],delta[0],delta[2]],np.float32)
            np.testing.assert_array_equal(m[pos],expected)
        finally:
            env.close()
    for x in states.values():
        np.testing.assert_array_equal(x[0],states["nominal"][0])
    residual = {k:float(np.max(abs(v-states["nominal"]))) for k,v in states.items()}
    angular = {k:float(np.max(abs(v[:,3:9]-states["nominal"][:,3:9]))) for k,v in states.items()}
    assert angular["translation"] > angular["translation_wrist"]
    np.testing.assert_allclose(states["translation_wrist"],states["nominal"],
                               atol=residual["repeat"]+np.finfo(np.float32).eps,rtol=0)
    return dict(numerical_max_abs_error=records, commands_per_real_trajectory=len(tape), real_cell="goal_2/15",
                max_robot_state_error=residual, max_rot6d_error=angular, renderer=gl,
                real_speed_max_abs_error={k:float(np.max(abs(v-speeds["nominal"]))) for k,v in speeds.items()},
                canonical_query_only_eef_position_changed=True)


def run(out, mode):
    import torch
    from reground.eval import jobs
    from reground.policies.vla import VLA
    import reference as ref
    import symmeta_adapter as stream
    import retry

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and not torch.cuda.is_available()
    started = time.monotonic()
    grid = retry.schedule()
    keys = next(b for b in grid if {k[0] for k in b} == {"goal_1","goal_2"})
    fact = dict(host=socket.gethostname(), code=os.environ["RETRY_PIN"], python=sys.version, gpu_hours=0,
                mode=mode, registered_batches=len(grid), unique_query_starts=len(set(k for b in grid for k in b)),
                released_labels=list(ref.SEEDS), executed_scientific_rows=0)
    if mode == "frames":
        fact.update(reciprocal())
    else:
        class NoisePolicy:
            revision, device, chunk_size, noise_dim = "cpu-noise-fixture", "cpu", 50, 32
            _noise = VLA._noise
            def predict_chunk(self, obs, tasks, seeds, q):
                from symmeta_cpu import processed
                commands = self._noise(seeds,q).numpy()[:,:,:7].copy()*.12
                for i,o in enumerate(obs):
                    state = processed(o)["observation.state"][0,:3].numpy()
                    commands[i,:,:3] += state*.3
                    commands[i,:,0] += .18
                return commands

        factory = retry.fresh
        ctx = jobs.WorkerCtx("pi05","cpu",len(keys),print)
        if mode == "real-policy":
            torch.set_num_threads(min(6,len(os.sched_getaffinity(0))))
            keys = [("goal_2",15,0)]
            ref.strict_policy(ctx)
            @contextmanager
            def short_check(work, stage, n):
                with retry.fresh(work,stage,n) as (pool, meter):
                    call = pool.call
                    def capped(idx,op,args):
                        result = call(idx,op,args)
                        if op=="reset" and stage in retry.ARMS:
                            for row in result:
                                row["max_steps"] = 11  # CPU wiring only, exercise the live requery
                        return result
                    pool.call = capped
                    yield pool,meter
            factory = short_check
        else:
            ctx._vla = NoisePolicy()
        try:
            work = out/"pair"
            work.mkdir()
            with stream.reader_process() as (pipe, reader):
                retry.run_pair(ctx.vla,pipe,reader,keys,work,factory)
            retry.validate(work,keys,ctx.vla.revision,native_horizon=mode!="real-policy")
            support = ref.read(work/"support.json")
            fact.update(keys=keys, support=[s["support"] for s in support],
                        estimates=[s["estimate"] for s in support], contracts=ref.read(work/"contract.json"),
                        query={arm: {k: v for k,v in ref.read(work/(arm+".json")).items()
                               if k in ("budget","rows","opening_sha256","speed_limit")} for arm in retry.ARMS},
                        samples_and_all_opening_leaves_checked=True)
            # Independently compare the two arms on the shared proposal, including angular deltas.
            tr = [jobs.load_traces(work/arm) for arm in retry.ARMS]
            changed = []
            for i,k in enumerate(keys):
                a,b = [t[k]["A"][:10] for t in tr]
                np.testing.assert_array_equal(a[:,[0,1,2,5,6]],b[:,[0,1,2,5,6]])
                changed.append(float(np.max(abs(a[:,3:5]-b[:,3:5]))))
            assert max(changed)>0
            fact["max_opening_wrist_change_by_lane"] = changed
        finally:
            ctx.close()
    fact["elapsed_seconds"] = time.monotonic()-started
    ref.write(out/"passed.json",fact)
    print(f"PASS {mode}: {out/'passed.json'}", flush=True)
