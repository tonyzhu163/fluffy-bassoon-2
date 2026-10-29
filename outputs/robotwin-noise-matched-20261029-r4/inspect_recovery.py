import hashlib,json,zipfile
from pathlib import Path
import torch
def same_json(a,b): return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
root=Path("/mowenjie/fluffy-bassoon-2/outputs/robotwin-noise-matched-20261029-r4/experiment")
summary=json.loads((root/"summary.json").read_text())
joint=torch.load(root/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
records={}
for method in ("ppo","sac","ppo_reset"):
    state=joint["states"][method]
    learner=state["learner"]
    individual=torch.load(root/(method+"-recovery.pt"),map_location="cpu",weights_only=False)
    record={"source_steps":state["source_steps"],"updates":learner["updates"],
            "pending_rollouts":len(state["pending_rollouts"]),
            "source_counter_matches_summary":state["source_steps"]==summary["source"][method]["source_steps"],
            "individual_counter_matches":individual["source_steps"]==state["source_steps"],
            "schedule_matches":individual["schedule"]==state["schedule"],
            "task_steps_match":state["task_steps"]==summary["source"][method]["task_steps"],
            "contract_matches":individual["contract"]==joint["contract"] and same_json(joint["contract"],summary["contract"]),
            "sampler_present":"rng" in state["schedule"],"learner_keys":sorted(learner)}
    if method=="sac":
        record.update(replay_rows=len(learner["replay"]),observed_chunks=learner["environment_steps"],
            optimizer_state_counts={name:len(learner[name]["state"]) for name in ("actor_optimizer","critic_optimizer","temperature_optimizer")},
            private_rng_present=all(name in learner for name in ("numpy_rng","torch_rng","train_torch_rng")),
            critic_target_present="target_critic" in learner)
    else:
        record.update(optimizer_steps=learner["optimizer_steps"],optimizer_state_count=len(learner["optimizer"]["state"]),
            pending_primitive_steps=sum(item[1]["primitive_steps"] for item in state["pending_rollouts"]),
            rollout_tensor_shapes=[{key:list(value.shape) for key,value in row.items()} for row,result in state["pending_rollouts"]],
            global_rng_present=sorted(learner["rng"]))
    records[method]=record
files={}
for name in ("coordinator-recovery.pt","ppo-recovery.pt","sac-recovery.pt","ppo_reset-recovery.pt"):
    path=root/name
    with path.open("rb") as stream: digest=hashlib.file_digest(stream,"sha256").hexdigest()
    with zipfile.ZipFile(path) as z: bad=z.testzip()
    files[name]={"bytes":path.stat().st_size,"sha256":digest,"zip_crc_ok":bad is None}
motor=json.loads((root/"frozen-motor.json").read_text())
result={"inspection":"read-only CPU checkpoint metadata and CRC; no learner instantiation, forward/backward, optimizer step or CUDA call",
        "status":summary["status"],"methods":records,"files":files,
        "next_method_index":joint["next_method_index"],"coordinator_rng_keys":sorted(joint["rng"]),
        "frozen_motor_unchanged":motor["unchanged"],"frozen_motor":motor,
        "global_contract_matches":same_json(joint["contract"],summary["contract"])}
print(json.dumps(result,indent=2))
