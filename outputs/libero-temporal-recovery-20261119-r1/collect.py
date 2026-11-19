"""Cross two persistent bottle faults with the same three query continuations."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
import json
from pathlib import Path
import sys

spec=spec_from_file_location("bottle_runtime",Path(__file__).resolve().parents[1] /
                             "libero-witness-transfer-20261119-r1/replay.py")
bottle=module_from_spec(spec)
spec.loader.exec_module(bottle)

import numpy as np
from scipy.spatial.transform import Rotation


visual=bottle.visual
ROOT=visual.ROOT
REACTIVE=ROOT/"outputs/libero-interrupted-practice-20261119-r1/release02/episodes"
SUPPORT=ROOT/"outputs/libero-witness-transfer-20261119-r1/cpu01"
MODES=("support","default","regrasp","transport")


def correction(step, tape, state, close):
    # All timing is set by the recorded nominal plan. The controller uses the
    # public current wrist to return to its approach after the 10 cm retreat.
    cut=close+12
    start=max(close-8,0)
    if step<cut:
        return tape["issued"][step]
    age=step-cut
    if age>=60:
        return tape["issued"][min(start+age-60,299)]
    target=tape["states"][max(0,start-1)].copy()
    if age<30:
        target[:3]=tape["states"][cut-1,:3]+[0,0,.10]
    dr=(Rotation.from_quat(target[3:7])*Rotation.from_quat(state[3:7]).inv()).as_rotvec()
    return np.r_[.65*np.clip((target[:3]-state[:3])/.05,-1,1),
                 .65*np.clip(dr/.5,-1,1),-1.].astype(np.float32)


def execute(pose,goal,fault,mode,path,predicates,bank,out):
    if mode=="transport":
        directory=REACTIVE/f"p{pose}-g{goal}-D18"
        rec=json.loads((directory/"result.json").read_text())
        assert (rec["pose"],rec["goal"],rec["mode"])==(pose,goal,"D18")
        with np.load(directory/"tape.npz") as z:
            tape={k:z[k] for k in ("issued","states","hits")}
    else:
        tape,rec,directory=bottle.read_tape(pose,goal,"sham" if mode=="support" else "query")
    close=None
    if mode=="regrasp":
        windows=np.lib.stride_tricks.sliding_window_view(tape["issued"][:,-1]>0,8)
        close=int(np.flatnonzero(windows.all(axis=1))[0])
    env=visual.LiberoEnvironmentAdapter()
    env._env=visual.ControlEnv(str(path),use_camera_obs=False,has_offscreen_renderer=True,camera_names=[])
    env._init_states=bank
    env._max_steps=300
    try:
        reset=env.reset(pose,18180100+pose)
        physics=reset["environment_reproducibility"]["initial_state_fingerprint"]
        assert physics==rec["physics"]
        world=env._env.env
        state=visual.robot_state(reset["observation"])
        initial_state=visual.digest(state)
        frames=[visual.images(world.sim)]
        start_rgb=visual.digest(frames[0])
        assert "llvmpipe" in visual.GL.glGetString(visual.GL.GL_RENDERER).decode()
        assert not any(all(world._eval_predicate(t) for t in p) for p in predicates)
        issued,states,hits=[],[],[]
        for step in range(300):
            a=correction(step,tape,state,close) if mode=="regrasp" else tape["issued"][step]
            resp=env.step(a.copy())
            state=visual.robot_state(resp["observation"])
            hit=[all(world._eval_predicate(t) for t in p) for p in predicates]
            assert resp["success"]==hit[0]
            issued.append(a); states.append(state); hits.append(hit)
            if mode=="support" and (step+1)%25==0:
                frames.append(visual.images(world.sim))
        log=world.perturbation_runtime.diagnostics()
        assert log["case_id"]==fault
        eventname="recoverable_grasp_failure" if fault=="D17" else "dynamic_state_change"
        events=[x for x in log["events"] if x["event"]==eventname]
        if events and fault=="D18":
            assert events[0]["target"]=="wine_bottle_1"
        issued,states,hits,frames=map(np.stack,(issued,states,hits,frames))
        if mode=="transport" and fault=="D18":
            np.testing.assert_array_equal(tape["states"],states)
            np.testing.assert_array_equal(tape["hits"],hits)
        if mode=="support":
            if fault=="D18":
                with np.load(SUPPORT/f"p{pose}-g{goal}-D18.npz") as old:
                    for k,x in (("states",states),("frames",frames),("issued",issued),("hits",hits)):
                        np.testing.assert_array_equal(old[k],x)
            np.savez_compressed(out/f"p{pose}-g{goal}-{fault}.npz",frames=frames,states=states,issued=issued,hits=hits)
        return dict(pose=pose,goal=goal,fault=fault,mode=mode,physics=physics,start_rgb=start_rgb,
                    start_state=initial_state,commands=visual.digest(issued),hit=bool(hits[:,goal].any()),
                    final=bool(hits[-1,goal]),other_goal=bool(hits[:,1-goal].any()),fired=bool(events),
                    event=events,diagnostics=log["events"],regrasp_close_control=close,
                    source=str(directory),controls=300)
    finally:
        env.close()


def pair(job):
    pose,goal,out=job
    _,predicates,bank=bottle.archive.setup("live")
    manifest=visual.TaskManifest.load(str(visual.RELEASE/"assets/task_manifest.json"))
    rows=[]
    for fault in ("D17","D18"):
        path=manifest.resolve_bddl("libero_goal",2,fault)
        for mode in MODES:
            rows.append(execute(pose,goal,fault,mode,path,predicates,bank,out))
    for key in ("physics","start_state","start_rgb"):
        assert len({r[key] for r in rows})==1
    for mode in ("support","default","transport"):
        assert len({r["commands"] for r in rows if r["mode"]==mode})==1
    return rows


if __name__=="__main__":
    out=Path(sys.argv[1])
    with (out/"branches.jsonl").open("x",buffering=1) as log, ProcessPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(pair,product(range(24,30),range(2),[out])):
            for row in rows:print(json.dumps(row),file=log)
            print("verified pose and goal",rows[0]["pose"],rows[0]["goal"],flush=True)
