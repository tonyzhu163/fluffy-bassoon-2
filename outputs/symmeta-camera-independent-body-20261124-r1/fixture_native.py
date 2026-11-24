"""One Kinova3 fixture, real released layout and numerical native interface, zero episodes."""
import argparse
import os
from pathlib import Path

import numpy as np

from bank import CASES_SHA, cases, sha
from native_panel import ARMS, commands, native, parse, release_rows
from fluffy_bassoon.xemb.embodiment import apply_embodiment_action, get_embodiment_spec, retarget_action_from_motor
from paired_records import checked, atomic, encode


def run(bank, bank_sha, resets_file, out):
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and os.sched_getaffinity(0)=={36,37}
    bankdata=parse(checked(bank,bank_sha))
    assert bank.name=="bank1.json" and bankdata["cases_sha256"]==CASES_SHA
    r,=bankdata["rows"]
    c=next(c for c in cases() if c["index"]==8)
    assert r["index"]==8 and r["archive_sha256"]==sha(r["archive_npz"])==c["support_sha256"]
    d=parse(checked(bank.parent/r["id"]/"diagnostics.json",bankdata["diagnostic_sha256"][r["id"]]))
    qr,=[v for v in parse(resets_file.read_bytes()) if v["shard"]==c["shard"] and v["reset"]["init_state_id"]==c["query_layout"]]
    ref=release_rows()[c["suite"],c["task_id"],c["query_layout"]]
    assert qr["reset"]["seed"]==c["query_seed"] and qr["horizon"]==ref["horizon"]
    assert qr["model"]["arm"]=="Kinova3" and qr["model"]["joint_count"]==7
    assert qr["model"]["cameras"]==["agentview","robot0_eye_in_hand"]
    assert all(qr["reset"][k]==ref[k] for k in ("released_row_fingerprint","bddl_sha256"))
    assert native().CHUNK_SIZE==5
    spec=get_embodiment_spec(c["embodiment"])
    truth=dict(yaw_deg=spec.yaw_deg,xy_scale=spec.xy_scale,gripper_sign=spec.gripper_sign)
    low,high=-np.ones(7),np.ones(7)
    with np.load(r["archive_npz"]) as z:
        means=z["issued_mean"][:3]
        held=np.repeat(means,5,axis=0)
        np.testing.assert_array_equal(held,d["issued_commands"])
        np.testing.assert_array_equal(apply_embodiment_action(held,spec,low,high),d["physical_commands"])
    # Independent numerical audit of the chunk inverse and singleton forward map,
    # including a non-unit gain and gripper flip. No model or simulator query.
    for probe in (truth,dict(yaw_deg=c["angle_deg"],xy_scale=.73,gripper_sign=-1)):
        issued,physical=commands(means,probe,truth)
        inverse=retarget_action_from_motor(means,probe["yaw_deg"],probe["xy_scale"],probe["gripper_sign"],low,high)
        np.testing.assert_array_equal(issued,inverse)
        np.testing.assert_array_equal(physical,[apply_embodiment_action(a,spec,low,high) for a in inverse])
    atomic(out/"native-interface.json",encode(dict(physical_id=r["id"],source_sha256=r["archive_sha256"],
           query_layout=c["query_layout"],query_seed=c["query_seed"],released_row_fingerprint=ref["released_row_fingerprint"],
           model_class=qr["model"]["model_class"],joint_count=7,planned_native_arms=list(ARMS),
           numerical_decoder_probes=2,held_commands=15,reset_reference_sha256=sha(resets_file),native_episodes=0)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("bank",type=Path)
    p.add_argument("bank_sha")
    p.add_argument("resets",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.bank,a.bank_sha,a.resets,a.out)
