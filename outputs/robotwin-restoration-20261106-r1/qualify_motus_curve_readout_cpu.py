"""Linux code/reference qualification, or later qualification of real C6/C10 inputs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace,MethodType
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from motus_curve import configuration as training_configuration
from motus_curve_readout import selected_steps,allowed_steps,original_sft_restore,result_steps,verify_one_restore
from motus_source_readout import configuration,ReadoutRollout,observe_native_ODE
from case_schedule import CaseSchedule
from optimizer_semantics import semantic_digest
from profile_checkpoint import verify
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
CONTRACTS=('MOTUS-SOURCE-GRPO-CURVEDEV06-CONTRACT.json','MOTUS-SOURCE-GRPO-CURVEDEV10-CONTRACT.json')


def require_rejected(call):
    try:call()
    except (AssertionError,ValueError):return
    raise AssertionError('Invalid endpoint or source-schedule input accepted')


def source_sft_references(plan):
    hashes={}
    for rank in range(8):
        for ordinal in range(6):
            path=Path(plan['source_SFT_reference_state'])/'sampling'/f'rank-{rank:02d}'/f'{ordinal:05d}.pt.gz'
            row=read_tensor(path)
            assert row['mode']=='eval' and row['producing_policy_iteration']==0
            assert row['rank']==rank and row['prediction_ordinal']==ordinal
            assert not row['terminal_value_only'] and row['forward_sampling_fields']['action'].shape==(2,448)
            hashes[str(path)]=sha(path)
    return hashes


def actual_inputs(plan):
    steps=selected_steps(plan);training=Path(plan['independent_training_readback_result'])
    assert sha(training)==plan['independent_training_readback_sha256']
    readback=json.loads(training.read_bytes())
    assert readback['status']=='passed_independent_native_GRPO_fresh_collection_curve_readback'
    assert readback['stop_iteration']==steps[-1] and readback['final_Adam_updates_per_rank']==15*steps[-1]
    assert not readback['learning_gain_claim']
    source_cfg=json.loads(Path(plan['original_config']).read_bytes())
    assert source_cfg['native_pilot']['stop_after_iteration']==steps[-1]
    cfg=configuration(plan,Path(os.environ['TASK_LOCAL'])/'readout-input-schema')
    checked={}
    for step in steps:
        receipt=plan['checkpoints'][str(step)];path=Path(receipt['path'])
        assert sha(path/'MANIFEST.json')==receipt['manifest_sha256']
        manifest=json.loads((path/'MANIFEST.json').read_bytes());assert manifest['global_step']==step
        verify(path,manifest)
        if step:
            assert any(r['step']==step and r['manifest_sha256']==receipt['manifest_sha256']
                and r['path']==str(path) for r in readback['full_checkpoints'])
        for rank in range(8):
            layout=json.loads((path/'pilot'/f'motus-optimizer-layout-{rank:02d}.json').read_bytes())
            signature=json.loads((path/'pilot'/f'motus-actor-signature-{rank:02d}.json').read_bytes())
            assert layout['optimizer_steps']==signature['optimizer_steps']==15*step
            assert semantic_digest(layout)==signature['optimizer_semantic_sha256']
            state=json.loads((path/'pilot'/f'motus-case-schedule-env-{rank:02d}.json').read_bytes())
            assert state['next_reset_ordinal']==4*step
            expected=(cfg.native_pilot.prefix_case_manifest_sha256 if step==0 else cfg.native_pilot.case_manifest_sha256)
            assert state['manifest_sha256']==expected
            for role in ('actor','rollout','env'):
                side=path/'pilot'/f'{role}-{rank:02d}.pt'
                saved=torch.load(side,map_location='cpu',weights_only=False)
                assert saved['role']==role and saved['rank']==rank and saved['rng']['cuda']
                assert saved['producing_policy_iteration']==max(0,step-1)
                if role=='actor':assert saved['optimizer_steps']==15*step and saved['optimizer_boundary']
                if role=='env':assert len(saved['samplers'])==2
                del saved
        checked[str(step)]=dict(path=str(path),manifest_sha256=receipt['manifest_sha256'],worker_sidecars=24)
    return dict(actual_endpoint_inputs_verified=True,checkpoints=checked,
        independent_training_readback_sha256=sha(training),original_training_config_sha256=sha(plan['original_config']),
        all48_original_SFT_sampling_files=source_sft_references(plan),
        actual_GPU_restore_and_SFT_calibration_still_required=True)


def schema_reference(plans):
    import rlinf.config as native
    import rlinf.models.embodiment.openpi.pi0 as pinned
    class Placement:
        def get_world_size(self,role):return 8
    training=json.loads((ROOT/'MOTUS-CURVE-GRPO-TRAIN01-CONTRACT.json').read_bytes())
    source=training_configuration(training)
    # This is explicitly a schema fixture, not a fabricated trained checkpoint.
    with tempfile.TemporaryDirectory(prefix='rt-curve-readout-schema-') as directory:
        fixture=Path(directory)/'training-config-schema.json'
        fixture.write_text(json.dumps(OmegaConf.to_container(source,resolve=True)))
        configs=[]
        for plan in plans:
            step=plan['selected_training_iteration']
            assert plan['status']=='prepared' and not plan['budget_extension_approved']
            assert selected_steps(plan)==(0,step)
            cfg=configuration(dict(plan,original_config=str(fixture)),Path(directory)/f'dev-{step}')
            assert cfg.runner.resume_dir is None and cfg.native_pilot.maximum_actions==32768
            assert cfg.native_pilot.readout_reference_root==plan['source_SFT_reference_state']
            assert cfg.native_pilot.readout_reference_root!=plan['source_state']
            assert allowed_steps(cfg)==(0,step)
            for key in ('algorithm','actor','rollout','env'):
                assert OmegaConf.to_container(cfg[key],resolve=True)==OmegaConf.to_container(source[key],resolve=True)
            with patch.object(native,'Cluster',lambda *a,**k:None),patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
                native.validate_cfg(cfg)
            original=json.loads((Path(plan['checkpoints']['0']['path'])/'pilot/motus-case-schedule-env-00.json').read_bytes())
            schedule=CaseSchedule(ROOT/'MOTUS-CURVE-GRPO-CASES.json',sha(ROOT/'MOTUS-CURVE-GRPO-CASES.json'))
            assert original_sft_restore(cfg,schedule,original,Path(directory)/'global_step_0/pilot')
            require_rejected(lambda:original_sft_restore(cfg,schedule,dict(original,next_reset_ordinal=1),Path(directory)/'global_step_0/pilot'))
            require_rejected(lambda:original_sft_restore(cfg,schedule,dict(original,manifest_sha256='0'*64),Path(directory)/'global_step_0/pilot'))
            assert not original_sft_restore(cfg,schedule,original,Path(directory)/'global_step_2/pilot')
            for bad in (dict(plan,extended_source_curve_readout=False),dict(plan,bank='confirmation'),dict(plan,algorithm='ppo'),dict(plan,selected_training_iteration=2)):
                require_rejected(lambda:selected_steps(bad))
            result=dict(checkpoints=[0,step],extended_source_curve_readout=True,algorithm='grpo',bank='dev',
                fixed_DEV_only=True,new_confirmation=False,input_checkpoints=plan['checkpoints'])
            assert result_steps(result)==[0,step]
            require_rejected(lambda:result_steps(dict(result,bank='confirmation')))
            require_rejected(lambda:result_steps(dict(result,extended_source_curve_readout=False)))
            fake=SimpleNamespace(action_horizon=32,action_dim=32,
                build_prefix_cache=lambda observation:(None,None,None),
                run_suffix=lambda observation,x,t,cache,mask:x*.125,
                velocity_from_suffix=lambda suffix:suffix)
            observation=SimpleNamespace(state=torch.zeros(2,14))
            worker=SimpleNamespace(cfg=cfg,_rank=0,_flush_pilot_io=lambda:None,
                hf_model=SimpleNamespace(named_parameters=lambda:iter(())))
            draws=[]
            with patch.object(pinned.model,'preprocess_observation',lambda value,train=False:value):
                for policy in (0,step):
                    ReadoutRollout.readout_begin(worker,policy,2)
                    expected=pinned.Pi0.sample_actions(fake,observation,num_steps=5);expected_rng=torch.get_rng_state()
                    ReadoutRollout.readout_begin(worker,policy,2)
                    actual,noise=observe_native_ODE(MethodType(pinned.Pi0.sample_actions,fake),32,observation,num_steps=5)
                    assert torch.equal(actual,expected) and torch.equal(expected_rng,torch.get_rng_state())
                    draws.append(noise)
            assert torch.equal(*draws)
            configs.append(dict(step=step,native_B2_preserved=True,actual_pinned_ODE_loop_reference=True,
                deterministic_suffix_stub=True,output_and_final_CPU_RNG_exact=True,paired_initial_draws_exact=True,
                full_model_GPU_still_required=True,schema_fixture_not_actual_trained_checkpoint=True))
            print('CURVE_READOUT_SCHEMA_AND_ODE_REFERENCE',step,'passed',flush=True)
        for step in (1,2):assert selected_steps(dict(selected_training_iteration=step,checkpoints={'0':{},str(step):{}}))==(0,step)
    legacy_plan=json.loads((ROOT/'MOTUS-SOURCE-GRPO-DEV02-CONTRACT.json').read_bytes())
    result=json.loads((BASE/'motus-source-grpo-dev02/state/RESULT.json').read_bytes())
    restores=[verify_one_restore(row['step'],row,legacy_plan['checkpoints'][str(row['step'])]) for row in result['restores']]
    print('CURVE_READOUT_ACTUAL_C0_C2_COLD_RESTORE_AUDIT passed',flush=True)
    return dict(configurations=configs,all48_original_SFT_sampling_files=source_sft_references(plans[0]),
        actual_existing_GPU_C0_and_C2_restore_records_rechecked_against_full_cold_sources=restores,
        original0_1_2_selection_preserved=True,unflagged_or_confirmation6_10_rejected=True,
        actual_checkpoint6_and10_inputs_missing_and_not_qualified=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--identity',default='cpumotuscurvereadout01');parser.add_argument('--contract',choices=CONTRACTS)
    args=parser.parse_args()
    assert re.fullmatch(r'cpumotuscurvereadout[0-9]{2}',args.identity)
    assert os.uname().sysname=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    code=ROOT.parents[1];source=os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==source
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    out=BASE/args.identity;out.mkdir()
    report=dict(status='checking',source_commit=source,GPU_seconds=0,models_or_simulators_constructed=False,
        GPU_readout_not_admitted=True)
    write(out/'RESULT.json',report)
    try:
        # Later endpoints must not depend on rewriting a previously admitted
        # endpoint's immutable contract after code advances on main.
        names=(args.contract,) if args.contract else CONTRACTS
        plans=[json.loads((ROOT/name).read_bytes()) for name in names]
        for plan in plans:
            for name,expected in plan['source_files'].items():assert sha(code/name)==expected,name
            assert sha(ROOT/'motus_source_readout.py')==plan['entry_sha256']
            assert sha(ROOT/'motus_source_readout.sh')==plan['launcher_sha256']
        subprocess.run(['bash','-n',str(ROOT/'motus_source_readout.sh')],check=True)
        if args.contract:
            plan=json.loads((ROOT/args.contract).read_bytes())
            assert plan['status']=='prepared'
            report.update(actual_inputs(plan),status='passed_actual_fixed_curve_DEV_checkpoint_input_CPU_qualification')
        else:
            report.update(schema_reference(plans),status='passed_fixed_curve_DEV_code_schema_and_pinned_ODE_CPU_reference')
        assert not torch.cuda.is_initialized()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:write(out/'RESULT.json',report)
    print('CURVE_READOUT_CPU_QUALIFIED',str(out/'RESULT.json'),sha(out/'RESULT.json'),flush=True)


if __name__=='__main__':main()
