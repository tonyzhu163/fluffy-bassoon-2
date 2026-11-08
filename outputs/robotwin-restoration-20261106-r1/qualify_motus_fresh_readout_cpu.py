"""Linux CPU qualification of fresh-C10 routing, reset arguments and native ODE."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from case_schedule import CaseSchedule
from motus_curve import configuration as training_configuration
from motus_curve_readout import selected_steps, allowed_steps, original_sft_restore, result_steps
from motus_fresh_readout import qualified_bank, result_bank, install_readout_reset, FRESH_FIELDS
from motus_source_readout import configuration, selected_seeds, ReadoutRollout, observe_native_ODE
from readback_motus_source_readout import verify_scene_bindings
from qualify_motus_curve_readout_cpu import schema_reference, actual_inputs, require_rejected
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
CONTRACT = 'MOTUS-SOURCE-GRPO-FRESHCONFIRMATION01-CONTRACT.json'


def fresh_reference(plan):
    import rlinf.config as native
    import rlinf.models.embodiment.openpi.pi0 as pinned
    assert plan['status'] == 'prepared' and not plan['budget_extension_approved']
    bank = qualified_bank(plan)
    assert selected_steps(plan) == (0, 10)
    for bad in (dict(plan, bank='confirmation'), dict(plan, bank='dev'),
            dict(plan, fresh_confirmation_readout=False), dict(plan, selected_training_iteration=6),
            dict(plan, algorithm='ppo'), dict(plan, extended_source_curve_readout=False)):
        require_rejected(lambda: selected_steps(bad))
    for key in ('fresh_confirmation_bank_sha256', 'fresh_confirmation_protocol_sha256'):
        bad = dict(plan, **{key: '0'*64})
        require_rejected(lambda: qualified_bank(bad))
    result = dict(algorithm='grpo', bank='fresh_confirmation', checkpoints=[0,10],
        input_checkpoints=plan['checkpoints'], selected_training_iteration=10,
        extended_source_curve_readout=True, fixed_DEV_only=False, new_confirmation=True,
        **{key: plan[key] for key in FRESH_FIELDS})
    assert result_steps(result) == [0,10] and result_bank(result) == bank
    require_rejected(lambda: result_steps(dict(result, bank='confirmation')))
    require_rejected(lambda: result_steps(dict(result, checkpoints=[0,6])))
    require_rejected(lambda: result_bank(dict(result, new_confirmation=False)))

    class Placement:
        def get_world_size(self, role): return 8

    train_plan = json.loads((ROOT/'MOTUS-CURVE-GRPO-TRAIN01-CONTRACT.json').read_bytes())
    source = training_configuration(train_plan)
    reset_calls, draw_count = [], 0
    with tempfile.TemporaryDirectory(prefix='rt-fresh-readout-schema-') as directory:
        fixture = Path(directory)/'training-schema.json'
        fixture.write_text(json.dumps(OmegaConf.to_container(source, resolve=True)))
        cfg = configuration(dict(plan, original_config=str(fixture)), Path(directory)/'fresh-output')
        assert cfg.runner.resume_dir is None and allowed_steps(cfg) == (0,10)
        assert cfg.native_pilot.readout_bank == 'fresh_confirmation'
        for key in ('algorithm','actor','rollout','env'):
            assert OmegaConf.to_container(cfg[key],resolve=True) == OmegaConf.to_container(source[key],resolve=True)
        with patch.object(native, 'Cluster', lambda *a, **k: None), patch.object(native,
                'HybridComponentPlacement', lambda *a, **k: Placement()):
            native.validate_cfg(cfg)
        original = json.loads((Path(plan['checkpoints']['0']['path'])/
            'pilot/motus-case-schedule-env-00.json').read_bytes())
        schedule = CaseSchedule(ROOT/'MOTUS-CURVE-GRPO-CASES.json', sha(ROOT/'MOTUS-CURVE-GRPO-CASES.json'))
        assert original_sft_restore(cfg, schedule, original, Path(directory)/'global_step_0/pilot')
        require_rejected(lambda: original_sft_restore(cfg, schedule,
            dict(original, next_reset_ordinal=1), Path(directory)/'global_step_0/pilot'))
        require_rejected(lambda: original_sft_restore(cfg, schedule,
            dict(original, manifest_sha256='0'*64), Path(directory)/'global_step_0/pilot'))
        assert not original_sft_restore(cfg, schedule, original, Path(directory)/'global_step_10/pilot')
        fake = SimpleNamespace(action_horizon=32, action_dim=32,
            build_prefix_cache=lambda obs:(None,None,None),
            run_suffix=lambda obs,x,t,cache,mask:x*.125,
            velocity_from_suffix=lambda suffix:suffix)
        obs = SimpleNamespace(state=torch.zeros(2,14))
        shared_draws = {}
        for step in (0,10):
            for panel in range(4):
                for rank in range(8):
                    seeds = selected_seeds(bank,panel,rank)
                    owner = SimpleNamespace(_readout_selected=seeds)
                    inventory = torch.arange(150)
                    env = SimpleNamespace(num_envs=2, reset_state_ids=torch.zeros(2,dtype=torch.long),
                        success_seeds=inventory,corrupt=False,
                        venv=SimpleNamespace(envs=[SimpleNamespace(env_seed=None,
                            task=SimpleNamespace(ep_num=None)) for _ in range(2)]))
                    def native_reset(env_idx=None, env_seeds=None):
                        reset_calls.append(dict(step=step,panel=panel,rank=rank,
                            indices=env_idx,seeds=list(env_seeds)))
                        for index,seed in zip(range(2) if env_idx is None else env_idx,env_seeds,strict=True):
                            env.venv.envs[index].env_seed=seed
                            env.venv.envs[index].task.ep_num=seed+int(env.corrupt)
                        return tuple(env_seeds)
                    env.reset = native_reset
                    install_readout_reset(owner,env)
                    assert env.reset() == tuple(seeds) and env.reset_state_ids.tolist() == seeds
                    assert env.reset(env_idx=[1,0], env_seeds=[-1,-1]) == (seeds[1],seeds[0])
                    assert env.reset(env_idx=[1]) == (seeds[1],)
                    assert env.success_seeds is inventory and torch.equal(inventory,torch.arange(150))
                    worker = SimpleNamespace(cfg=cfg,_rank=rank,_flush_pilot_io=lambda:None,
                        hf_model=SimpleNamespace(named_parameters=lambda:iter(())))
                    with patch.object(pinned.model,'preprocess_observation',lambda value,train=False:value):
                        ReadoutRollout.readout_begin(worker,step,panel)
                        expected = [pinned.Pi0.sample_actions(fake,obs,num_steps=5) for _ in range(6)]
                        expected_rng = torch.get_rng_state()
                        ReadoutRollout.readout_begin(worker,step,panel)
                        for chunk in range(6):
                            actual,noise = observe_native_ODE(MethodType(pinned.Pi0.sample_actions,fake),
                                32,obs,num_steps=5)
                            assert torch.equal(actual,expected[chunk])
                            key = panel,rank,chunk
                            if step == 0: shared_draws[key] = noise
                            else: assert torch.equal(shared_draws[key],noise)
                            draw_count += 1
                        assert torch.equal(expected_rng,torch.get_rng_state())
        env.corrupt=True
        require_rejected(lambda:env.reset())
    assert draw_count == 384 and len(reset_calls) == 193
    legacy_cases=json.loads((ROOT/'MOTUS-SOURCE-READOUT-CASES.json').read_bytes())['dev']
    actual_bindings=verify_scene_bindings(BASE/'motus-source-grpo-dev03/state',[0,1],legacy_cases)
    return dict(actual_qualified_bank_bound=True,bank_sha256=plan['fresh_confirmation_bank_sha256'],
        bank_cases=64,old_confirmation_unflagged_and_non_C10_rejected=True,
        actual_C0_schedule_migration_guard_passed=True,unchanged_native_config_sections=True,
        reset_argument_reference_calls=192,substituted_physical_seed_rejected=True,
        actual_old_GPU_scene_bindings_rechecked=actual_bindings,sampler_inventory_preserved=True,
        native_pinned_ODE_draws=draw_count,paired_batch_draws_tensor_exact=192,
        native_ODE_output_and_final_CPU_RNG_exact=True,deterministic_suffix_stub=True,
        physical_reset_and_full_model_GPU_readout_still_required=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', required=True)
    parser.add_argument('--actual-inputs', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'cpumotusfreshreadout[0-9]{2}',args.identity)
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code,source = ROOT.parents[1],os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip() == source
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    plan = json.loads((ROOT/CONTRACT).read_bytes())
    out = BASE/args.identity
    out.mkdir()
    report = dict(status='checking',source_commit=source,GPU_seconds=0,
        model_or_simulator_constructed=False,GPU_readout_admitted=False)
    write(out/'RESULT.json',report)
    try:
        names = (CONTRACT,) if args.actual_inputs else (
            'MOTUS-SOURCE-GRPO-CURVEDEV06-CONTRACT.json',
            'MOTUS-SOURCE-GRPO-CURVEDEV10-CONTRACT.json',CONTRACT)
        plans = [json.loads((ROOT/name).read_bytes()) for name in names]
        for item in plans:
            for name,expected in item['source_files'].items(): assert sha(code/name) == expected,name
            assert sha(ROOT/'motus_source_readout.py') == item['entry_sha256']
            assert sha(ROOT/'motus_source_readout.sh') == item['launcher_sha256']
        subprocess.run(['bash','-n',str(ROOT/'motus_source_readout.sh')],check=True)
        if args.actual_inputs:
            assert selected_steps(plan) == (0,10)
            qualified_bank(plan)
            report.update(actual_inputs(plan),status='passed_actual_fresh_C10_checkpoint_inputs_CPU_qualification')
        else:
            report.update(fresh_reference(plan),legacy_fixed_DEV_reference=schema_reference(plans[:2]),
                actual_C10_checkpoint_inputs_qualified=False,
                status='passed_fresh_C10_code_schema_reset_and_pinned_ODE_CPU_reference')
        assert not torch.cuda.is_initialized()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally: write(out/'RESULT.json',report)
    h=sha(out/'RESULT.json')
    print('FRESH_READOUT_CPU_QUALIFIED',str(out/'RESULT.json'),*[h[i:i+16] for i in range(0,64,16)],flush=True)


if __name__ == '__main__': main()
