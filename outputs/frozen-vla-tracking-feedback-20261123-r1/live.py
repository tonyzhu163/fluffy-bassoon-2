"""Separate fresh native starts for a source-learned motor feedback test."""
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "frozen-vla-native-transfer-20261123-r1"
sys.path.insert(0, str(OLD))
from panel import TASKS, SUITES, key, save
from native_rollout import Trace, check_native
from ur5e_motor import run_episode, load_frozen_pi05
from ur5e_sim_worker import UR5eSimulator
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from feedback import Feedback, FeedbackTrace

MATRICES = Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/cpu01/matrices.json")

def cases():
    for suite,t in TASKS:
        for layout in (16,17):
            yield dict(arm="Kinova3",suite=suite,task_id=t,init_state_id=layout,variant="BASE",
                       seed=41800000 + 10000*SUITES.index(suite) + 100*t + layout)


def rows(sim,policy,case,contract,matrix,max_controls=None):
    for mode in ("raw","static","feedback","stale"):
        trace = Trace(sim,np.array(matrix,dtype=np.float32) if mode == "static" else None)
        wrapper = (FeedbackTrace(trace,Feedback(contract,1 if mode == "feedback" else 3))
                   if mode in ("feedback","stale") else trace)
        rec = run_episode(wrapper,policy,case,get_embodiment_spec("nom"),max_controls=max_controls)
        if max_controls is None:
            check_native(rec)
            if trace.first["renderer_vendor"] != "NVIDIA Corporation":
                raise ValueError("Expected NVIDIA EGL on admitted device")
        n = sum(len(c["cmd"]) for c in trace.chunks)
        if n != rec["controls"]:
            raise ValueError("Incomplete native command tape")
        if mode == "raw":
            reference = rec
            opening_chunks = trace.chunks[:3]
        else:
            if rec["reset"] != reference["reset"] or rec["opening_proposal_sha256"] != reference["opening_proposal_sha256"]:
                raise ValueError("Unmatched robot reset, native noise or frozen initial proposal")
            if mode != "static" and trace.chunks[:3] != opening_chunks:
                raise ValueError("Feedback arms differ before their first eligible intervention")
        corrections = wrapper.corrections if isinstance(wrapper,FeedbackTrace) else None
        yield mode,dict(case=case,mode=mode,record=rec,model=trace.first["model_identity"],
                        full_reset=trace.first["environment_reproducibility"],chunks=trace.chunks,corrections=corrections)


def main(out):
    model_path=HERE/"source-model.json"
    encoded=model_path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != "19607168a6ff7425f461db94211b49b09b0bb6259ca896559a5d1ead6e2432e9":
        raise ValueError("Unexpected source response and persistence fit")
    source=json.loads(encoded)
    matrix=json.loads(MATRICES.read_bytes())
    if hashlib.sha256(MATRICES.read_bytes()).hexdigest() != "20a9d6800e10404ee23a542c70a3ba1fb077daeccd56cf7fea93555c7add64cf":
        raise ValueError("Wrong fixed comparator calibration")
    policy=load_frozen_pi05()
    with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
        sim=UR5eSimulator()
    summary=[]
    with sim:
        for case in cases():
            m=matrix[f"Kinova3/{case['suite']}/{case['task_id']}"]
            if m["support"]["start"] !=11 or m["support"]["controls"]!=15:
                raise ValueError("Unexpected support budget or layout")
            result={}
            for mode,body in rows(sim,policy,case,source,m["matrix"]):
                save(out/f"{key(case)}-{mode}.json",body)
                r=body["record"]
                result[mode]=dict(success=r["success"],native_return=r["native_return"],controls=r["controls"])
            summary.append(dict(case=case,results=result))
            print(key(case),result,flush=True)
    save(out/"summary.json",dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),rows=summary))


if __name__=="__main__":
    main(Path(sys.argv[1]))
