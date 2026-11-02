"""Compile a verified terminal PPO source into separate steering AD/DPT views."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np

from fluffy_bassoon.meta_rl.frozen_steering_history import (
    SteeringSource, checked_reference, load_episode, save_corpus, split_streams,
)
from fluffy_bassoon.meta_rl.vla_history import file_sha256


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
    config=dict(b['source']);config['tasks']=tuple(config['tasks']);source=SteeringSource(**config)
    count=source.episodes_per_task_per_update
    if count%8 or b['validation_slots']!=sorted(np.random.default_rng(90211).choice(count,count//8,replace=False).tolist()):
        raise ValueError('Preserve the declared seed90211 whole-stream12.5% validation split')
    episodes=[load_episode(ref,source) for ref in b['episode_checkpoints']]
    streams=split_streams(episodes,source,validation_slots=b['validation_slots'],
                          expected_updates=tuple(b['expected_updates']))
    provenance=dict(campaign=b['campaign'],source_binding=dict(path=str(a.binding.resolve()),
        sha256=file_sha256(a.binding)),completion_evidence=b['completion_evidence'],
        source_episode_references=b['episode_checkpoints'],
        train_validation_rule='Whole task/collection-slot streams split before labels/windows',
        validation_slots=b['validation_slots'],expected_updates=b['expected_updates'])
    manifests=save_corpus(a.output,streams,source,teacher_reference=b['teacher'],provenance=provenance)
    datasets={key:dict(path=str((a.output/(key.replace('_','-')+'.npz')).resolve()),
        sha256=value['sha256'],transitions=value['transitions'],trials=value['trials'])
        for key,value in manifests.items()}
    assembly=dict(completed=True,campaign=b['campaign'],source=asdict(source),datasets=datasets,
        teacher=b['teacher'],source_binding=provenance['source_binding'],
        physical_episodes=len(episodes),full_physical_history=True,
        source_physical_horizons=sorted({int(item['physical_episode']['horizon']) for item in episodes}),
        PPO_prefix_and_raw_suffix_retained=True,non_target_history_identical=True)
    (a.output/'assembly.json').write_text(json.dumps(assembly,indent=2)+'\n')
    print(json.dumps(assembly,indent=2))


if __name__=='__main__':main()
