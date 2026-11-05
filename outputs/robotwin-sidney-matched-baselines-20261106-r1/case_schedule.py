"""Immutable source-case layout shared by the fresh PPO/GRPO comparison."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree,tensor_hash,write
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import save_tensor


class CaseSchedule:
    def __init__(self,path,expected_sha256):
        path=Path(path);raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected_sha256
        self.sha256=expected_sha256;self.data=json.loads(raw)
        assert self.data['task']=='place_phone_stand'
        assert self.data['iterations']==2 and self.data['rounds_per_iteration']==4
        assert self.data['environment_group_size']==8 and len(self.data['groups'])==256
        assert len({row['scene_seed'] for row in self.data['groups']})==256

    def rows(self,ordinal,rank):
        assert 0<=ordinal<8 and 0<=rank<4
        iteration,round_index=divmod(ordinal,4)
        rows=sorted((r for r in self.data['groups'] if r['policy_iteration']==iteration
            and r['round_index']==round_index and r['rank']==rank),key=lambda r:r['local_group'])
        assert len(rows)==8 and [r['local_group'] for r in rows]==list(range(8))
        assert all(r['replicas']==8 for r in rows)
        return rows


def exposed_articulation_state(task):
    return [dict(qpos=a.get_qpos().tolist(),qvel=a.get_qvel().tolist(),
        root_p=a.get_root_pose().p.tolist(),root_q=a.get_root_pose().q.tolist(),
        root_linear_velocity=a.get_root_linear_velocity().tolist(),
        root_angular_velocity=a.get_root_angular_velocity().tolist())
        for a in task.scene.get_all_articulations()]


def install(owner,physical,schedule,root):
    original=physical.reset
    def reset(env,env_idx=None,env_seeds=None):
        # The declared source loop has one full reset per round. No partial
        # reset or autoreset may silently change the shared case chronology.
        assert env_idx is None and not env.auto_reset
        ordinal=owner._paired_reset_ordinal
        rows=schedule.rows(ordinal,owner._rank)
        seeds=[row['scene_seed'] for row in rows for _ in range(8)]
        assert len(seeds)==env.num_envs==64
        env.reset_state_ids=torch.tensor(seeds,dtype=torch.long,device=env.reset_state_ids.device)
        result=original(env_idx=None,env_seeds=seeds);obs=result[0]
        receipt=[];raw=[]
        for row in rows:
            start=row['local_group']*8;stop=start+8
            subs=env.venv.envs[start:stop]
            seeds_match=all(int(s.env_seed)==int(s.task.ep_num)==row['scene_seed'] for s in subs)
            prompts=list(obs['task_descriptions'][start:stop])
            scene=subs[0].task.get_info();physics=exposed_articulation_state(subs[0].task)
            scene_match=physics_match=True;observation_match=True;differences=[]
            for index,sub in enumerate(subs,start):
                current_scene=sub.task.get_info();current_physics=exposed_articulation_state(sub.task)
                scene_match=scene_match and current_scene==scene
                physics_match=physics_match and current_physics==physics
                error={}
                for key in ('main_images','wrist_images','states'):
                    left,right=obs[key][index],obs[key][start]
                    same=torch.equal(left,right);observation_match=observation_match and same
                    error[key]=dict(exact=same,max_abs=float((left.double()-right.double()).abs().max()),
                        unequal_elements=int((left!=right).sum()))
                differences.append(dict(env_index=index,public_observation=error,
                    exposed_articulation_exact=current_physics==physics,scene_exact=current_scene==scene))
                raw.append(dict(env_index=index,requested_seed=int(sub.env_seed),actual_seed=int(sub.task.ep_num),
                    scene_info=current_scene,exposed_articulations=current_physics))
            passed=seeds_match and len(set(prompts))==1 and scene_match and physics_match and observation_match
            receipt.append(dict(**row,actual_scene_info=scene,task_description=prompts[0],
                initial_public_observation_sha256=tensor_hash({k:obs[k][start]
                    for k in ('main_images','wrist_images','states')}),
                exposed_articulation_sha256=tensor_hash(physics),
                complete_native_physics_snapshot=False,replica_count=8,passed=passed,
                seeds_match=seeds_match,prompt_identity=len(set(prompts))==1,
                scene_identity=scene_match,articulation_identity=physics_match,
                public_observation_identity=observation_match,differences=differences))
        # Persist every actual replica before checking identity, including raw
        # state/physics needed to diagnose a failed exact-equality gate.
        raw_receipt=save_tensor(root/'group-resets'/f'rank-{owner._rank:02d}-reset-{ordinal:04d}.pt',dict(
            manifest_sha256=schedule.sha256,public_observation=cpu_tree(obs),replicas=raw))
        passed=all(r['passed'] for r in receipt)
        write(root/'group-resets'/f'rank-{owner._rank:02d}-reset-{ordinal:04d}.json',dict(
            manifest_sha256=schedule.sha256,reset_ordinal=ordinal,rank=owner._rank,groups=receipt,
            raw_replicas=raw_receipt,passed=passed,
            group_public_reset_identity=all(r['public_observation_identity'] for r in receipt),
            group_exposed_articulation_identity=all(r['articulation_identity'] for r in receipt)))
        assert passed,'Common scene-group identity failed; raw replicas persisted before gate'
        owner._paired_reset_ordinal+=1
        return result
    physical.reset=MethodType(reset,physical)


def boundary_ordinal(directory,rounds_per_iteration=4):
    # Only whole-iteration recovery is supported. It needs no in-flight native
    # physics restoration, and the fixed four-round contract determines ordinal.
    checkpoint=Path(directory).parent
    step=int(checkpoint.name.removeprefix('global_step_'))
    assert step in (0,1,2)
    assert rounds_per_iteration in (1,4)
    return rounds_per_iteration*step
