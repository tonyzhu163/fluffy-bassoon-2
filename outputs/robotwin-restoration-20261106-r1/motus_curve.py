"""Explicit continuation of the verified two-cycle GRPO prefix to iteration10."""
import hashlib
import json
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended_schedule():
    prefix_path=ROOT/'MOTUS-TRAIN-CASES.json'
    prefix=json.loads(prefix_path.read_bytes())
    bank=json.loads((ROOT/'MOTUS-PHONE-TRAIN-SEED-BANK.json').read_bytes())
    seeds=list(bank['success_seeds'])
    assert len(seeds)==len(set(seeds))==1000
    random.Random(0).shuffle(seeds)
    assert [r['scene_seed'] for r in prefix['groups']]==seeds[:256]
    groups=[]
    for ordinal in range(1280):
        iteration,offset=divmod(ordinal,128)
        round_index,group=divmod(offset,32)
        rank,local=divmod(group,8)
        groups.append(dict(policy_iteration=iteration,round_index=round_index,
            global_group=group,rank=rank,local_group=local,
            scene_seed=seeds[ordinal%1000],replicas=8))
    assert groups[:256]==prefix['groups']
    assert prefix['train_bank_sha256']==bank['full_source_sha256']
    return dict(format='motus_source_curve_cases_v1',task='place_phone_stand',
        status='existing_native_TRAIN_bank_no_new_simulator_qualification',
        iterations=10,rounds_per_iteration=4,environment_group_size=8,
        global_groups_per_round=32,case_group_occurrences=1280,
        unique_scenes=1000,replicas_per_group=8,episodes_per_arm=10240,
        prefix_iterations=2,prefix_manifest_sha256=file_sha(prefix_path),
        train_bank_sha256=bank['full_source_sha256'],
        selection='Retain Random0 native TRAIN permutation; consume all1000 scenes before cyclic repeats; no outcome filtering.',
        groups=groups)


def segment_action_limit(start,stop):
    assert (start,stop) in ((2,6),(6,10))
    # Four fresh1024-episode collections and five16-episode progress panels.
    return (stop-start)*1024*192+(stop-start+1)*16*192


def validate_curve(cfg):
    from omegaconf import OmegaConf
    from motus_configuration import validate
    start=int(cfg.native_pilot.segment_start_iteration)
    stop=int(cfg.native_pilot.stop_after_iteration)
    assert (start,stop) in ((2,6),(6,10))
    assert cfg.native_pilot.extended_source_curve is True
    assert cfg.native_pilot.intended_iterations==10
    assert cfg.runner.max_steps==cfg.runner.max_epochs==10
    assert cfg.native_pilot.keep_local_checkpoints==2
    ordinals=list(cfg.native_pilot.curve_reference_first_ordinals)
    assert len(ordinals)==8 and all(isinstance(n,int) and n>=0 for n in ordinals)
    assert Path(cfg.native_pilot.curve_reference_root).name=='sampling'
    assert cfg.native_pilot.maximum_actions==segment_action_limit(start,stop)
    assert cfg.native_pilot.comparison_algorithm=='grpo'
    assert cfg.native_pilot.require_behavior_likelihood_match and int(cfg.native_pilot.world_size)==8
    assert Path(cfg.runner.resume_dir).name==f'global_step_{start}'
    path=Path(cfg.native_pilot.case_manifest_path)
    assert file_sha(path)==cfg.native_pilot.case_manifest_sha256
    schedule=json.loads(path.read_bytes())
    assert schedule==extended_schedule(), 'Continuation must preserve the exact two-cycle prefix and committed rotation'
    prefix=ROOT/'MOTUS-TRAIN-CASES.json'
    assert cfg.native_pilot.prefix_case_manifest_sha256==file_sha(prefix)==schedule['prefix_manifest_sha256']
    native=json.loads(Path(cfg.env.train.seeds_path).read_bytes())['place_phone_stand']['success_seeds']
    assert file_sha(cfg.env.train.seeds_path)==schedule['train_bank_sha256']
    assert set(native)==set(r['scene_seed'] for r in schedule['groups']) and len(native)==1000
    # Validate every unchanged numerical/motor/runtime field through the
    # existing two-cycle guard after checking the explicit changed fields above.
    projected=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    projected.native_pilot.extended_source_curve=False
    projected.runner.max_steps=projected.runner.max_epochs=2
    projected.native_pilot.stop_after_iteration=2
    projected.native_pilot.maximum_actions=500000
    projected.native_pilot.case_manifest_path=str(prefix)
    projected.native_pilot.case_manifest_sha256=file_sha(prefix)
    summary=validate(projected)
    return dict(summary,intended_iterations=10,segment_start_iteration=start,
        admitted_stop_after_iteration=stop,maximum_actions=segment_action_limit(start,stop),
        train_manifest_sha256=cfg.native_pilot.case_manifest_sha256,
        prefix_train_manifest_sha256=file_sha(prefix),original_two_cycle_prefix_preserved=True,
        new_collections=stop-start,new_episodes=(stop-start)*1024,
        optimizer_steps_at_start=15*start,optimizer_steps_at_stop=15*stop,
        constant_optimizer_schedule_preserved=True)


def configuration(plan):
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(json.loads(Path(plan['original_config']).read_bytes()))
    start,stop=plan['start_iteration'],plan['stop_iteration']
    assert (start,stop) in ((2,6),(6,10))
    local=Path(os.environ['TASK_LOCAL']);cold=Path(os.environ['TASK_COLD'])
    cfg.runner.max_steps=cfg.runner.max_epochs=10
    cfg.runner.resume_dir=str(local/'resume'/f'global_step_{start}')
    cfg.runner.logger.log_path=str(local/'logs')
    cfg.runner.logger.experiment_name=plan['output_identity']
    cfg.runner.per_worker_log_path=str(cold/'state/workers')
    cfg.native_pilot.output=str(cold/'state')
    cfg.native_pilot.accounting_root=str(cold/'action-accounting')
    cfg.native_pilot.attempt=plan['output_identity']
    cfg.native_pilot.deadline_unix=int(os.environ['TASK_DEADLINE'])
    cfg.native_pilot.extended_source_curve=True
    cfg.native_pilot.intended_iterations=10
    cfg.native_pilot.segment_start_iteration=start
    cfg.native_pilot.stop_after_iteration=stop
    cfg.native_pilot.maximum_actions=segment_action_limit(start,stop)
    cfg.native_pilot.keep_local_checkpoints=2
    cfg.native_pilot.passive_controller_source=os.environ['TASK_CONTROLLER']
    cfg.native_pilot.passive_controller_sha256=os.environ['TASK_CONTROLLER_SHA']
    cfg.native_pilot.checkpoint_archive=str(cold/'checkpoints')
    for kind,name in [('case','MOTUS-CURVE-GRPO-CASES.json'),('dev','MOTUS-DEV-CASES.json')]:
        cfg.native_pilot[f'{kind}_manifest_path']=str(ROOT/name)
        cfg.native_pilot[f'{kind}_manifest_sha256']=file_sha(ROOT/name)
    cfg.native_pilot.prefix_case_manifest_sha256=file_sha(ROOT/'MOTUS-TRAIN-CASES.json')
    cfg.native_pilot.curve_reference_root=plan['resume_dev_sampling_root']
    cfg.native_pilot.curve_reference_first_ordinals=plan['resume_dev_sampling_first_ordinals']
    validate_curve(cfg)
    return cfg


if __name__=='__main__':
    import sys
    assert sys.argv[1:] == ['write-static-manifest']
    (ROOT/'MOTUS-CURVE-GRPO-CASES.json').write_text(json.dumps(extended_schedule(),indent=2)+'\n')
