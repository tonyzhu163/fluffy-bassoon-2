"""Actual Linux CPU qualification; no model/simulator construction or GPU admission."""
import argparse
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

import motus_curve_entry
import motus_teacher_entry as teacher
from motus_curve import configuration,extended_schedule,segment_action_limit,validate_curve
from case_schedule import CaseSchedule,boundary_ordinal
from optimizer_semantics import semantic_digest
from profile_checkpoint import verify
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def native_loop_reference(cfg,start,stop):
    """Execute the pinned native loop with RPC stubs; qualify routing, not SGD."""
    events=[];steps=dict(actor=None,rollout=None,env=None);current={'batch':None,'adv':None,'optimizer':15*start}
    class Done:
        def __init__(self,value=None):self.value=value
        def wait(self):return self.value
    class Actor:
        def set_global_step(self,step):steps['actor']=step;return Done()
        def sync_model_to_rollout(self):return Done()
        def recv_rollout_trajectories(self,**unused):
            assert current['batch']==steps['actor'];return Done()
        def compute_advantages_and_returns(self):
            assert current['batch']==steps['actor']
            current['adv']=current['batch'];return Done([{}])
        def run_training(self):
            assert current['adv']==current['batch']==steps['actor']
            assert current['optimizer']==15*steps['actor']
            current['optimizer']+=15;events.append(('update',steps['actor']));return Done([{}])
    class Rollout:
        def set_global_step(self,step):steps['rollout']=step;return Done()
        def sync_model_from_actor(self):return Done()
        def generate(self,**unused):
            assert len(set(steps.values()))==1 and steps['actor'] is not None
            assert current['batch']!=steps['actor'], 'A fresh collect must replace the previous batch'
            current['batch']=steps['actor'];events.append(('fresh_collection',steps['actor']));return Done()
    class Environment:
        def set_global_step(self,step):steps['env']=step;return Done()
        def interact(self,**unused):return Done()
    runner=SimpleNamespace(cfg=cfg,global_step=start,max_steps=stop,actor=Actor(),rollout=Rollout(),env=Environment(),
        reward=None,overlap_env_bootstrap=False,weight_sync_interval=1,env_channel=None,rollout_channel=None,
        actor_channel=None,reward_channel=None,timer=lambda *a,**k:nullcontext(),
        _should_profile_step=lambda step:False,_maybe_eval_and_checkpoint=lambda step:{},
        _log_step_metrics=lambda **kwargs:None,_finish_run=lambda:None)
    runner.update_rollout_weights=lambda:teacher.EmbodiedRunner.update_rollout_weights(runner)
    teacher.EmbodiedRunner.run(runner)
    assert runner.global_step==stop and current['optimizer']==15*stop
    assert events==[event for step in range(start,stop) for event in [('fresh_collection',step),('update',step)]]
    return dict(start=start,stop=stop,events=events,final_optimizer_counter=current['optimizer'],
        native_loop_executed=True,RPC_stubs_only=True,actual_model_resume_still_required=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--identity',default='cpumotuscurve01')
    args=parser.parse_args()
    assert __import__('re').fullmatch(r'cpumotuscurve[0-9]{2}',args.identity)
    assert os.uname().sysname=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    code=ROOT.parents[1];source=os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==source
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    out=BASE/args.identity;out.mkdir()
    report=dict(status='checking',source_commit=source,GPU_seconds=0,models_or_simulators_constructed=False,
        GPU_training_not_admitted=True,actual_model_resume_and_new_collection_still_required=True)
    write(out/'RESULT.json',report)
    try:
        plan=json.loads((ROOT/'MOTUS-CURVE-GRPO-TRAIN01-CONTRACT.json').read_bytes())
        assert plan['status']=='prepared' and not plan['budget_extension_approved']
        for name,expected in plan['source_files'].items():assert sha(code/name)==expected,name
        assert sha(ROOT/'motus_curve_entry.py')==plan['entry_sha256']
        assert sha(ROOT/'train_motus_curve01.sh')==plan['launcher_sha256']
        subprocess.run(['bash','-n',str(ROOT/'train_motus_curve01.sh')],check=True)
        prefix=Path(plan['resume_checkpoint']['path'])
        assert sha(prefix/'MANIFEST.json')==plan['resume_checkpoint']['manifest_sha256']
        manifest=json.loads((prefix/'MANIFEST.json').read_bytes());assert manifest['global_step']==2
        verify(prefix,manifest)
        prior=Path(plan['independent_prefix_readback_result'])
        assert sha(prior)==plan['independent_prefix_readback_sha256']
        prior_report=json.loads(prior.read_bytes())
        assert prior_report['status']=='passed_independent_behavior_matched_training_readback'
        assert prior_report['completed_iterations']==2 and prior_report['Adam_updates_per_rank']==30
        assert prior_report['checked_json_hashes'][plan['original_config']]==sha(plan['original_config'])
        descriptors=[]
        for rank in range(8):
            layout=json.loads((prefix/'pilot'/f'motus-optimizer-layout-{rank:02d}.json').read_bytes())
            signature=json.loads((prefix/'pilot'/f'motus-actor-signature-{rank:02d}.json').read_bytes())
            assert layout['optimizer_steps']==signature['optimizer_steps']==30
            assert semantic_digest(layout)==signature['optimizer_semantic_sha256']
            state=json.loads((prefix/'pilot'/f'motus-case-schedule-env-{rank:02d}.json').read_bytes())
            assert state['next_reset_ordinal']==8 and state['manifest_sha256']==sha(ROOT/'MOTUS-TRAIN-CASES.json')
            for role in ('actor','rollout','env'):
                path=prefix/'pilot'/f'{role}-{rank:02d}.pt'
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['role']==role and saved['rank']==rank
                assert saved['producing_policy_iteration']==1
                assert saved['rng']['cuda'], 'Source GPU RNG state must be retained'
                if role=='actor':assert saved['optimizer_steps']==30 and saved['optimizer_boundary']
                if role=='env':assert len(saved['samplers'])==2
                descriptors.append(dict(role=role,rank=rank,sha256=sha(path),saved_policy_tag=saved['producing_policy_iteration']))
                del saved
        print('CURVE_CPU_PREFIX_STATE_VERIFIED all24 worker sidecars and all8 owned Adam descriptors',flush=True)
        reference_hashes={}
        from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
        for rank in range(8):
            rank_root=Path(plan['resume_dev_sampling_root'])/f'rank-{rank:02d}'
            first=plan['resume_dev_sampling_first_ordinals'][rank]
            expected=[rank_root/f'{first+offset:05d}.pt.gz' for offset in range(6)]
            assert sorted(rank_root.glob('*.pt.gz'))[-6:]==expected
            for offset in range(6):
                path=expected[offset]
                saved=read_tensor(path)
                assert saved['mode']=='eval' and saved['producing_policy_iteration']==2
                assert saved['prediction_ordinal']==first+offset and saved['rank']==rank and not saved['terminal_value_only']
                assert saved['forward_sampling_fields']['action'].shape==(2,448)
                reference_hashes[str(path)]=sha(path)
        print('CURVE_CPU_ALL48_SOURCE_DEV_REFERENCES_VERIFIED per-worker measured ordinals',flush=True)
        schedule=CaseSchedule(ROOT/'MOTUS-CURVE-GRPO-CASES.json',sha(ROOT/'MOTUS-CURVE-GRPO-CASES.json'))
        assert schedule.data==extended_schedule()
        prefix_rows=json.loads((ROOT/'MOTUS-TRAIN-CASES.json').read_bytes())['groups']
        assert schedule.data['groups'][:256]==prefix_rows
        seen=[]
        for ordinal in range(40):
            old=[r['scene_seed'] for rank in range(4) for r in schedule.rows(ordinal,rank,4)]
            new=[r['scene_seed'] for rank in range(8) for r in schedule.rows(ordinal,rank,8)]
            assert old==new and len(set(new))==32
            seen.extend(new)
        assert len(seen)==1280 and len(set(seen[:1000]))==1000 and seen[1000:]==seen[:280]
        for step in range(11):assert boundary_ordinal(Path('/tmp')/f'global_step_{step}'/'pilot',4)==4*step
        import rlinf.config as native
        class Placement:
            def get_world_size(self,role):return 8
        original=json.loads(Path(plan['original_config']).read_bytes())
        configs=[];loops=[]
        for start,stop in ((2,6),):
            cfg=configuration(plan)
            summary=validate_curve(cfg)
            current=OmegaConf.to_container(cfg,resolve=True)
            assert all(current[k]==original[k] for k in ('algorithm','actor','rollout','env'))
            for field,bad_value in [('actor.micro_batch_size',256),('actor.optim.lr',1e-4)]:
                bad=OmegaConf.create(current);OmegaConf.update(bad,field,bad_value)
                try:validate_curve(bad)
                except (AssertionError,ValueError):pass
                else:raise AssertionError('Changed numerical field accepted: '+field)
            with patch.object(native,'Cluster',lambda *a,**k:None),patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
                native.validate_cfg(cfg)
            assert segment_action_limit(start,stop)==801792
            configs.append(summary);loops.append(native_loop_reference(cfg,start,stop))
        # This checks only the pinned loop's later counters. A second segment
        # needs a new contract and measured per-rank references from actual C6.
        loops.append(native_loop_reference(cfg,6,10))
        print('CURVE_CPU_NATIVE_LOOP_VERIFIED fresh collect/update tags and counters',flush=True)
        # Preserve the original two-cycle config's validation path as well.
        legacy=OmegaConf.create(original)
        legacy.native_pilot.case_manifest_path=str(ROOT/'MOTUS-TRAIN-CASES.json')
        legacy.native_pilot.dev_manifest_path=str(ROOT/'MOTUS-DEV-CASES.json')
        assert teacher.validate(legacy)['intended_iterations']==2
        runner_path=Path(inspect.getsourcefile(teacher.EmbodiedRunner))
        storage={}
        for name,path in [('two_cycle_state',BASE/'motusbehaviorgrpo01/state'),('checkpoint2',prefix)]:
            raw=subprocess.check_output(['du','-sb',str(path)],text=True,timeout=180)
            storage[name+'_bytes']=int(raw.split()[0])
        probe=out/'write-probe'
        with probe.open('xb') as f:f.write(b'curve-Linux-CPU-preflight');f.flush();os.fsync(f.fileno())
        assert probe.read_bytes()==b'curve-Linux-CPU-preflight';probe.unlink()
        report.update(status='passed_continuation_code_prefix_and_native_loop_CPU_qualification',
            full_checkpoint2_cold_file_hashes=True,all48_source_checkpoint2_DEV_sampling_files_checked=reference_hashes,all24_source_worker_sidecars_checked=descriptors,
            all8_strict_owned_Adam_descriptors=True,source_policy_tag1_will_advance_to_fresh_policy2=True,
            original256_group_prefix_exact=True,all40_rounds_four_and_eight_rank_case_order_exact=True,
            all1000_TRAIN_scenes_before_repeat=True,configurations=configs,native_loop_references=loops,
            future_segment6_to10_requires_actual_checkpoint6_and_sampling_qualification=True,
            algorithm_actor_rollout_environment_fields_unchanged=True,
            numerical_LR_and_batch_mutations_rejected=True,original_two_cycle_guard_still_passes=True,
            native_runner_file=str(runner_path),native_runner_sha256=sha(runner_path),
            measured_storage=storage,BOS_write_read_fsync=True,
            filesystem_quota_not_exposed_by_virtual_BOS_df=True,live_storage_admission_still_required=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:write(out/'RESULT.json',report)
    print('CURVE_CPU_QUALIFIED',str(out/'RESULT.json'),sha(out/'RESULT.json'),flush=True)


if __name__=='__main__':main()
