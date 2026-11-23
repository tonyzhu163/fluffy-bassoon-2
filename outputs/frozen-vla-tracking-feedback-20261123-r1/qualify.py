"""CPU fixture on actual Kinova3 native resets; policy stub permits testing control indexing."""
import json
from pathlib import Path
import numpy as np
from live import cases, rows, MATRICES
from ur5e_sim_worker import UR5eSimulator


class Fixed:
    def infer(self, item, *, noise):
        if item["observation/state"].shape!=(8,) or noise.shape!=(10,32):
            raise ValueError("Unexpected frozen VLA schema")
        a=np.zeros((10,7),np.float32)
        a[:,:3]=np.tanh(noise[:,:3]*0.2 + [0.15,0.1,-0.15])
        return {"actions":a}


def main():
    root=Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-tracking-feedback-20261123-r1/cpu01")
    contract=json.loads((root/"result.json").read_bytes())
    matrices=json.loads(MATRICES.read_bytes())
    starts=[]
    with UR5eSimulator(allow_mesa=True) as sim:
        for case in cases():
            spec,first=sim.call("create",dict(case,embodiment="nom"))
            starts.append(dict(case=spec,physics=first["environment_reproducibility"]["initial_state_fingerprint"],
                               arm=first["model_identity"]["arm"]))
        case=next(cases())
        m=matrices[f"Kinova3/{case['suite']}/{case['task_id']}"]["matrix"]
        body=dict(rows(sim,Fixed(),case,contract,m,max_controls=25))
    acts={}
    for mode in ("feedback","stale"):
        deltas=np.concatenate(body[mode]["corrections"])
        assert deltas.shape==(25,7) and np.array_equal(deltas[:15],np.zeros((15,7)))
        assert not np.any(deltas[:,3:]) and np.any(deltas[15:,:3]!=0)
        cmds=np.concatenate([x["cmd"] for x in body[mode]["chunks"]])
        base=cmds-deltas
        assert np.all(np.abs(cmds)<=1) and np.all(np.linalg.norm(deltas[:,:3],axis=1)<=np.linalg.norm(base[:,:3],axis=1)+1e-6)
        acts[mode]=dict(first_changed_control=int(np.where(np.any(deltas!=0,axis=1))[0][0])+1,
                        l2_change=float(np.linalg.norm(deltas)),native_rewards=body[mode]["record"]["native_rewards"])
    assert not np.array_equal(body["feedback"]["corrections"],body["stale"]["corrections"])
    print(json.dumps(dict(fixture="25 controls per branch; synthetic action generator; native physics; no task success inference",
                          arms=["raw","static","feedback","stale"],starts=starts,action_checks=acts)))


if __name__=="__main__":
    main()
