"""Compile a verified terminal PPO source into separate steering AD/DPT views."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

from fluffy_bassoon.meta_rl.frozen_steering_history import (
    EpisodeSchedule, SteeringSource, case_identity, checked_reference, interface_record,
    load_episode, save_corpus, split_streams,
)
from fluffy_bassoon.meta_rl.vla_history import file_sha256


def prospective_schedule(reference, source, recipe):
    document=json.loads(checked_reference(reference).read_text())
    if (document.get('format')!='frozen_steering_episode_schedule_v1'
            or document.get('source_commit')!=source.source_commit
            or document.get('runtime_commit')!=source.runtime_commit):
        raise ValueError('Require the exact producer prospective episode schedule')
    versions=tuple(range(recipe['iterations']))
    schedule=EpisodeSchedule(document['entries'],source,expected_updates=versions)
    requested={};request_counts={version:0 for version in versions};references=set()
    shards=recipe.get('collection_shards',recipe.get('rollout_gpus_per_arm'))
    if type(shards) is not int or shards<1:raise ValueError('Bind actual producer request shards')
    for item in document['producer_requests']:
        identity=(item['path'],item['sha256'])
        if identity in references:raise ValueError('Duplicate producer request reference')
        references.add(identity)
        request=torch.load(checked_reference(item),map_location='cpu',weights_only=False)
        version=request['policy_iteration']
        if (request['phase']!='train' or version not in versions or request['iteration']!=version+1
                or request['source_commit']!=source.source_commit or request['runtime_commit']!=source.runtime_commit
                or request['lanes']!=recipe['envs_per_gpu']
                or interface_record(request['spec'])!=source.interface or not request['cases']):
            raise ValueError('Producer request changed finalized source/behavior/interface/execution contract')
        request_counts[version]+=1
        for case in request['cases']:
            key=(version,case_identity(case))
            if key in requested:raise ValueError('A prospective case occurs in two producer requests')
            requested[key]=identity
    if (set(requested)!=set(schedule.by_case) or any(count!=shards for count in request_counts.values())
            or any(sum(v==version for v,_ in requested)!=recipe['episodes_per_update'] for version in versions)):
        raise ValueError('Schedule must retain every prospective case in every finalized source bank')
    for version in versions:
        for task in source.tasks:
            cases=[json.loads(encoded) for v,encoded in requested if v==version and json.loads(encoded)['task']==task]
            if (not cases or any(type(case['slot']) is not int or case['slot']<0 for case in cases)
                    or len({case['slot'] for case in cases})!=len(cases)):
                raise ValueError('Every source bank needs unique prospective case slots for each task')
            for slot,case in enumerate(sorted(cases,key=lambda case:case['slot'])):
                if schedule.by_case[(version,case_identity(case))]!=(task,slot):
                    raise ValueError('Task streams must follow prospective case order, never outcomes')
    return schedule,requested


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux':raise ValueError('Data/model computation runs on allocated Linux')
    b=json.loads(a.binding.read_text())
    if b.get('format')!='steering_history_source_binding_v1' or b.get('source_ready') is not True:
        raise ValueError('Require independently verified terminal source binding, not a queued/running producer')
    if not b.get('completion_evidence'):
        raise ValueError('Retain actual scheduler/training/checkpoint completion evidence')
    for evidence in b['completion_evidence']:checked_reference(evidence)
    recipe=b.get('producer_recipe',{})
    if (recipe.get('finalized') is not True or type(recipe.get('iterations')) is not int
            or recipe['iterations']<2 or type(recipe.get('envs_per_gpu')) is not int
            or recipe['envs_per_gpu']<1 or not recipe.get('authority_reference')
            or b['expected_updates']!=list(range(recipe['iterations']))):
        raise ValueError('Require finalized human-agreed producer iterations/envs per GPU and exact behavior coverage')
    checked_reference(recipe['authority_reference'])
    config=dict(b['source']);config['tasks']=tuple(config['tasks']);source=SteeringSource(**config)
    if source.episodes_per_task_per_update!=recipe.get('episodes_per_task_per_update'):
        raise ValueError('Bind the finalized per-task source batch; no provisional16-episode default')
    schedule=requested=None
    if b.get('source_schedule'):
        schedule,requested=prospective_schedule(b['source_schedule'],source,recipe)
    if source.episodes_per_task_per_update is None and schedule is None:
        raise ValueError('Variable task counts require all prospective producer requests and cases')
    available=(schedule.slots if schedule is not None else {
        task:list(range(source.episodes_per_task_per_update)) for task in source.tasks})
    expected_validation={task:sorted(np.random.default_rng(90211).choice(slots,
        max(1,(len(slots)+7)//8),replace=False).tolist()) for task,slots in available.items()}
    declared_validation=(b['validation_slots'] if isinstance(b['validation_slots'],dict)
                         else {task:b['validation_slots'] for task in source.tasks})
    if declared_validation!=expected_validation:
        raise ValueError('Preserve seed90211 whole task/stream validation split, ceil one eighth per task')
    episodes=[load_episode(ref,source,schedule=schedule) for ref in b['episode_checkpoints']]
    if requested is not None:
        for episode in episodes:
            item=episode['request_reference'];key=(episode['update'],case_identity(episode['physical_episode']['case']))
            if item is None or (item['path'],item['sha256'])!=requested[key]:
                raise ValueError('Actual episode does not bind its prospective producer request')
    streams=split_streams(episodes,source,validation_slots=b['validation_slots'],
                          expected_updates=tuple(b['expected_updates']),schedule=schedule)
    provenance=dict(campaign=b['campaign'],source_binding=dict(path=str(a.binding.resolve()),
        sha256=file_sha256(a.binding)),completion_evidence=b['completion_evidence'],
        source_episode_references=b['episode_checkpoints'],
        train_validation_rule='Whole task/collection-slot streams split before labels/windows',
        validation_slots=b['validation_slots'],expected_updates=b['expected_updates'],producer_recipe=recipe,
        prospective_schedule=b.get('source_schedule'),
        validation_rule='Seed90211, ceil one eighth of each task stream set; selected before windows/labels')
    manifests=save_corpus(a.output,streams,source,teacher_reference=b['teacher'],provenance=provenance)
    datasets={key:dict(path=str((a.output/(key.replace('_','-')+'.npz')).resolve()),
        sha256=value['sha256'],transitions=value['transitions'],trials=value['trials'])
        for key,value in manifests.items()}
    assembly=dict(completed=True,campaign=b['campaign'],source=asdict(source),datasets=datasets,
        teacher=b['teacher'],source_binding=provenance['source_binding'],producer_recipe=recipe,
        physical_episodes=len(episodes),full_physical_history=True,
        source_physical_horizons=sorted({int(item['physical_episode']['horizon']) for item in episodes}),
        PPO_prefix_and_raw_suffix_retained=True,non_target_history_identical=True)
    (a.output/'assembly.json').write_text(json.dumps(assembly,indent=2)+'\n')
    print(json.dumps(assembly,indent=2))


if __name__=='__main__':main()
