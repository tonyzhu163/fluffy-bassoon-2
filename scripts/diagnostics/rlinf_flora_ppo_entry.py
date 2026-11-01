#!/usr/bin/env python
"""Task44 qualification/training entry; execution only on allocated Linux."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import RLINF,initializer_manifest,sha
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_config import NORM_SHA,evaluation_schedule


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['smoke','baseline','train'],required=True)
    parser.add_argument('--arm',choices=['S','F'],required=True)
    parser.add_argument('--world',type=int,choices=[1,2,4,8],required=True)
    for key in ('profile','model','norm','output','exchange','monitoring','accounting','attempt'):
        parser.add_argument('--'+key,required=True)
    args,overrides=parser.parse_known_args()
    if platform.system()!='Linux':raise RuntimeError('Run only inside the actual allocated Linux job')
    code=Path(__file__).resolve().parents[2]
    profile=json.loads(Path(args.profile).read_text())
    def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if head(code)!=profile['source_commit']:raise ValueError('Source deployment pin differs from admission')
    upstream=Path(os.environ['EMBODIED_PATH']).resolve().parents[1]
    if head(upstream)!=RLINF:raise ValueError('Use exact c605 official RLinf checkout')
    if sha(args.norm)!=NORM_SHA:raise ValueError('Use byte-identical bundled RLinf40 normalization')
    model=Path(args.model)
    if (model/'full_weights.pt').exists():raise ValueError('Do not shadow published source with RLinf full_weights.pt')
    loading_ref=profile['initializer_artifact_receipt']
    if sha(loading_ref['path'])!=loading_ref['sha256']:raise ValueError('Initializer receipt changed')
    loading=json.loads(Path(loading_ref['path']).read_text())
    declared=set()
    for row in initializer_manifest(loading,model):
        artifact=Path(row['path']).resolve()
        if not artifact.is_relative_to(model.resolve()) or sha(artifact)!=row['sha256']:
            raise ValueError('Published checkpoint artifact identity differs')
        declared.add(artifact)
    weights=set(model.resolve().glob('*.safetensors'))
    if not weights or not weights<=declared:raise ValueError('Every loaded safetensors shard must be receipt-bound')
    for key in ('model','norm','output','exchange','monitoring','accounting','profile'):
        if not Path(getattr(args,key)).is_absolute():raise ValueError('Explicit absolute resource paths required')
    schedule=evaluation_schedule(args.world,'DEV' if args.phase=='train' else args.phase)
    values=dict(ARM=args.arm,PHASE=args.phase,WORLD=args.world,PLACEMENT=f'0-{args.world-1}',
        EVAL_SLOTS=len(schedule),ONLY_EVAL=str(args.phase!='train').lower(),MODEL=args.model,NORM=args.norm,
        OUTPUT=args.output,EXCHANGE=args.exchange,MONITORING=args.monitoring,ACCOUNTING=args.accounting,
        PROFILE=args.profile,PROFILE_SHA=sha(args.profile),ATTEMPT=args.attempt)
    for key,value in values.items():os.environ['FLORA_PPO_'+key]=str(value)
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import actor_class,rollout_class
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_env import environment_class
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runner import MatchedRunner
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    actor_module.EmbodiedFSDPActor=actor_class()
    path=Path(os.environ['EMBODIED_PATH'])/'train_embodied_agent.py'
    spec=importlib.util.spec_from_file_location('flora_official_embodied_entry',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    entry.EmbodiedRunner=MatchedRunner;entry.EnvWorker=environment_class();entry.MultiStepRolloutWorker=rollout_class()
    sys.argv=[sys.argv[0],'--config-path',str(code/'configs/native_ppo'),'--config-name','libero44_flora',*overrides]
    entry.main()


if __name__=='__main__':main()
