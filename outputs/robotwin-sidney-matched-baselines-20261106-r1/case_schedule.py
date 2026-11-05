"""Immutable source-case layout shared by the fresh PPO/GRPO comparison."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash,write


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
        receipt=[]
        for row in rows:
            start=row['local_group']*8;stop=start+8
            subs=env.venv.envs[start:stop]
            assert all(int(s.env_seed)==int(s.task.ep_num)==row['scene_seed'] for s in subs)
            prompts=list(obs['task_descriptions'][start:stop]);assert len(set(prompts))==1
            scene=subs[0].task.get_info();physics=exposed_articulation_state(subs[0].task)
            for index,sub in enumerate(subs,start):
                assert sub.task.get_info()==scene
                assert exposed_articulation_state(sub.task)==physics
                for key in ('main_images','wrist_images','states'):
                    assert torch.equal(obs[key][index],obs[key][start]),f'Group reset differs: {key}/{index}'
            receipt.append(dict(**row,actual_scene_info=scene,task_description=prompts[0],
                initial_public_observation_sha256=tensor_hash({k:obs[k][start]
                    for k in ('main_images','wrist_images','states')}),
                exposed_articulation_sha256=tensor_hash(physics),
                complete_native_physics_snapshot=False,replica_count=8))
        write(root/'group-resets'/f'rank-{owner._rank:02d}-reset-{ordinal:04d}.json',dict(
            manifest_sha256=schedule.sha256,reset_ordinal=ordinal,rank=owner._rank,groups=receipt,
            group_public_reset_identity=True,group_exposed_articulation_identity=True))
        owner._paired_reset_ordinal+=1
        return result
    physical.reset=MethodType(reset,physical)


def boundary_ordinal(directory):
    # Only whole-iteration recovery is supported. It needs no in-flight native
    # physics restoration, and the fixed four-round contract determines ordinal.
    checkpoint=Path(directory).parent
    step=int(checkpoint.name.removeprefix('global_step_'))
    assert step in (0,1,2)
    return 4*step
