"""Resolve the unapproved Coffee proposal on Linux CPUs; never launch training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-coffee-ppo4-source-revision-20261105-r1'
CODE = ROOT / 'sources/source5-df425f40'
PIN = 'df425f4067299687daf1e3a914e1ebc24309c8e6'
VENDOR = ROOT / 'sources/critic-selected-a1229239-r1'


def main(*, reuse_existing=False):
    assert sys.platform == 'linux'
    assert json.loads((RUN / 'PROPOSAL.json').read_text())['status'] == 'proposal_only_not_approved'
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD']).decode().strip() == PIN
    output = RUN / 'RESOLVED-PROPOSAL-02.yaml'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=f'{CODE}/src:{VENDOR}', TASK_CODE_PIN=PIN, TASK_ATTEMPT='coffee4-40-r1',
        TASK_DEADLINE='1', RLINF_PATH=str(VENDOR), ROBOCASA_FLOW_TASK='CoffeeSetupMug',
        ROBOCASA_SIM_PPO_OUTPUT=str(RUN / 'source/attempt-01'),
        ROBOCASA_SIM_PPO_ACCOUNTING=str(RUN / 'accounting'),
        ROBOCASA_SIM_PPO_ACTOR_PLACEMENT='0-1', ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT='0-7',
        ROBOCASA_SIM_PPO_WORLD_SIZE='8', ROBOCASA_SIM_PPO_MICRO_BATCH='16',
        ROBOCASA_SIM_PPO_GLOBAL_ESCROW='true', ROBOCASA_FLOW_TRAIN_SEED='2500000',
        ROBOCASA_FLOW_EVAL_SEED='650100', ROBOCASA_FLOW_FRESH_EVAL_PROCESSES='true',
        ROBOCASA_FLOW_MAX_EPOCHS='40', ROBOCASA_FLOW_UPDATE_EPOCHS='4',
        ROBOCASA_FLOW_FULL_MODEL_EXPORT_STEPS='[40]', ROBOCASA_FLOW_VAL_INTERVAL='10',
        ROBOCASA_FLOW_SAVE_INTERVAL='10', ROBOCASA_FLOW_NOISE_LEVEL='0.3',
        ROBOCASA_FLOW_IGNORE_LAST='true', ROBOCASA_FLOW_TRAIN_ENVS='128',
        ROBOCASA_FLOW_ROUNDS='1', ROBOCASA_FLOW_MAX_ACTIONS='4100000',
        ROBOCASA_FLOW_SAMPLING_RECORD='full', ROBOCASA_FLOW_ACTOR_BATCH_DROP='[]',
        ROBOCASA_PI05_TORCH_MODEL=str(ROOT / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'),
        ROBOCASA_PI05_NATIVE_NORM=str(ROOT / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'))
    for key in ('ROBOCASA_SIM_PPO_RESUME', 'ROBOCASA_SIM_PPO_BEFORE_REFERENCE', 'ROBOCASA_SIM_PPO_BEFORE_SHA256'):
        env.pop(key, None)
    if reuse_existing:
        assert output.is_file()
    else:
        with output.open('xb') as stream:
            subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]robocasa_flow_ppo.py'),
                '--config-name', 'robocasa365_pi05_flow_sde_ppo_reuse4', '--cfg', 'job', '--resolve'],
                cwd=CODE, env=env, stdout=stream, check=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(CODE / 'src'), str(VENDOR)]
    from omegaconf import OmegaConf
    from rlinf.config import validate_fsdp_cfg
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_contract
    cfg = OmegaConf.load(output)
    cfg.actor = validate_fsdp_cfg(cfg.actor)
    contract = flow_contract(cfg)
    old_run = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
    reference = json.loads((old_run / 'source5/TurnOnSinkFaucet/attempt-01/flow-contract.json').read_text())
    reference.update(task='CoffeeSetupMug', train_seed=2500000, eval_seed=650100)
    assert contract == reference
    original = json.loads((old_run / 'source5/TurnOnSinkFaucet/attempt-01/resolved-config.json').read_text())
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert all(resolved[key] == original[key] for key in ('actor', 'rollout', 'algorithm'))
    assert cfg.runner.resume_dir is None and cfg.native_pilot.before_reference is None
    receipt = dict(passed=True, proposal_only_not_approved=True, GPU_hours=0, models_loaded=0,
        source_commit=PIN, config_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        recipe_exact_to_qualified_source5_except_task_and_registered_seeds=True,
        actor_rollout_algorithm_exact=True, contract=contract,
        qualification_scope='CPU composition and native actor FSDP validation; actual allocation/device checks remain required',
        deadline_note='Value1 is a non-executing config-resolution sentinel; an approved launch must bind its actual deadline')
    with (RUN / 'CPU-QUALIFICATION.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-existing', action='store_true')
    main(reuse_existing=parser.parse_args().validate_existing)
