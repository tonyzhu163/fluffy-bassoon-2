"""Explicit overrides for this run; no simulator/model work on the editing host."""
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def overrides(algorithm,qualification=False,resume=None):
    assert algorithm in ('ppo','grpo')
    assert not qualification or algorithm=='grpo'
    manifest=ROOT/'CASES.json'
    args=[f'algorithm.adv_type={"gae" if algorithm=="ppo" else "grpo"}',
        f'algorithm.loss_type={"actor_critic" if algorithm=="ppo" else "actor"}',
        f'actor.model.add_value_head={str(algorithm=="ppo").lower()}',
        'algorithm.group_size=8','env.train.group_size=8','actor.micro_batch_size=64',
        'algorithm.recompute_logprobs=false','native_pilot.memory_strategy=native_batch_end',
        'native_pilot.trim_after_scene_close=false',
        f'native_pilot.maximum_actions={120000 if qualification else 1000000}',
        'native_pilot.previous_maximum_actions=null',
        f'+native_pilot.comparison_algorithm={algorithm}',
        f'+native_pilot.qualification_only={str(qualification).lower()}',
        f'+native_pilot.case_manifest_path={manifest}',
        f'+native_pilot.case_manifest_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}',
        f'runner.logger.experiment_name=sidney-paired-{algorithm}']
    if qualification:
        args+=['runner.max_steps=1','runner.max_epochs=1',
            'env.train.rollout_epoch=1','algorithm.update_epoch=1']
    if resume:args.append(f'runner.resume_dir={resume}')
    return args


if __name__=='__main__':
    if sys.argv[1] in ('validate','validate_all'):
        import json
        import os
        import socket
        import torch
        from hydra import compose,initialize_config_dir
        sys.path.insert(0,str(ROOT.parent/'robotwin-sidney-rlinf-ppo-20261105-r1'))
        from paired_recipe import validate_pair
        if sys.argv[1]=='validate_all':
            assert sys.platform=='linux' and socket.gethostname()==sys.argv[2]
            assert socket.gethostname().startswith('job-') and os.environ['CUDA_VISIBLE_DEVICES']==''
            specs=[('ppo',False),('grpo',False),('grpo',True)]
        else:specs=[(sys.argv[2],sys.argv[3]=='qualification')]
        rows=[]
        for algorithm,qualification in specs:
            with initialize_config_dir(version_base='1.1',config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
                cfg=compose(config_name='sidney_phone_two',overrides=overrides(algorithm,qualification))
            rows.append(dict(algorithm=algorithm,qualification_only=qualification,
                native_world_value=cfg.native_pilot.world_size,
                native_world_type=type(cfg.native_pilot.world_size).__name__,
                contract=validate_pair(cfg)))
        assert not torch.cuda.is_initialized(),'Metadata-only preflight initialized CUDA'
        print(json.dumps(dict(status='complete',host=socket.gethostname(),configs=rows,
            model_or_simulator_constructed=False,cuda_initialized=False)))
        sys.exit(0)
    print('\n'.join(overrides(sys.argv[1],sys.argv[2]=='qualification',
        sys.argv[3] if len(sys.argv)>3 else None)))
