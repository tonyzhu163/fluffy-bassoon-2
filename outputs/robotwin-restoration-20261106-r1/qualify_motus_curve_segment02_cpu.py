"""Linux code qualification, or later qualification of the real C6 inputs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from motus_curve import configuration,validate_curve
from motus_curve_segment02_inputs import verify_inputs
from qualify_motus_curve_cpu import native_loop_reference
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def completion_guard_reference(launcher,plan):
    """Check the actual script's C6→10 closeout using disposable schema fixtures."""
    guard=launcher.read_text().rsplit("<<'PY'\n",1)[1].split('\nPY',1)[0]
    with tempfile.TemporaryDirectory(prefix='rt-segment02-closeout-schema-') as directory:
        root=Path(directory);out=root/'signals';state=root/'cold/state'
        out.mkdir();state.mkdir(parents=True)
        contract=root/'schema-contract.json';write(contract,plan)
        write(state/'RESULT.json',dict(status='completed_admitted_prefix',completed_iterations=10,
            intended_iterations=10,native_adam_updates_per_rank=150))
        write(state/'operational-restore.json',dict(workers=[{}]*24))
        write(state/'resume-baseline-reference.json',dict(all48_predictions_exact=True))
        for step in range(7,11):write(state/f'checkpoint-receipts/step-{step:04d}.json',dict(completed=True))
        for step in range(6,11):write(state/f'dev/iteration-{step:04d}.json',dict(completed=True))
        for index in range(128):write(state/f'group-resets/{index}.json',dict(passed=True))
        for index in range(32):
            write(state/f'learner-reference/{index}.json',dict(passed=True))
            write(state/f'behavior-reference/{index}.json',dict(original_behavior_exact=True))
        write(out/'resources.json',dict(sampled_reserve_passed=True))
        env=dict(os.environ,TASK_CURVE_PLAN=str(contract))
        passed=subprocess.run([os.sys.executable,'-',str(out),str(state)],input=guard,
            text=True,capture_output=True,env=env)
        assert passed.returncode==0,passed.stderr
        result=json.loads((out/'COMPLETE.json').read_bytes())
        assert result['checkpoint_boundaries']==[7,8,9,10] and result['native_adam_updates_per_rank']==150
        assert result['new_episodes']==4096 and not result['learning_improvement_claim']
        write(state/'case-migration/unexpected.json',dict())
        rejected=subprocess.run([os.sys.executable,'-',str(out),str(state)],input=guard,
            text=True,capture_output=True,env=env)
        assert rejected.returncode!=0 and 'AssertionError' in rejected.stderr
    return dict(actual_final_launcher_guard_executed=True,temporary_synthetic_receipts_only=True,
        C6_same_schedule_requires_no_second_migration=True,unexpected_migration_rejected=True)


def code_reference(plan):
    import rlinf.config as native
    first=json.loads((ROOT/'MOTUS-CURVE-GRPO-TRAIN01-CONTRACT.json').read_bytes())
    fixture=deepcopy(plan)
    fixture.update(original_config=first['original_config'],
        resume_dev_sampling_root=first['resume_dev_sampling_root'],
        resume_dev_sampling_first_ordinals=first['resume_dev_sampling_first_ordinals'])
    # This is a schema fixture using original configuration/reference locations.
    # No C6 model, state, prediction offsets or optimizer tensors are invented.
    cfg=configuration(fixture)
    summary=validate_curve(cfg)
    assert summary['optimizer_steps_at_start']==90 and summary['optimizer_steps_at_stop']==150
    original=json.loads(Path(first['original_config']).read_bytes())
    current=OmegaConf.to_container(cfg,resolve=True)
    assert all(current[key]==original[key] for key in ('algorithm','actor','rollout','env'))
    class Placement:
        def get_world_size(self,role):return 8
    with patch.object(native,'Cluster',lambda *a,**k:None),patch.object(native,
            'HybridComponentPlacement',lambda *a,**k:Placement()):
        native.validate_cfg(cfg)
    loop=native_loop_reference(cfg,6,10)
    assert loop['events']==[event for step in range(6,10) for event in
        [('fresh_collection',step),('update',step)]]
    try:verify_inputs(plan)
    except AssertionError:pass
    else:raise AssertionError('Prepared C6 placeholders were accepted as actual inputs')
    launcher=ROOT/'train_motus_curve02.sh'
    subprocess.run(['bash','-n',str(launcher)],check=True)
    # Execute the actual first Python admission block. A prepared contract must
    # fail before allocating devices, creating outputs or importing the learner.
    guard=launcher.read_text().split("<<'PY'\n",1)[1].split('\nPY',1)[0]
    blocked=subprocess.run([os.sys.executable,'-',str(ROOT/'MOTUS-CURVE-GRPO-TRAIN02-CONTRACT.json')],
        input=guard,text=True,capture_output=True)
    assert blocked.returncode!=0 and 'AssertionError' in blocked.stderr
    return dict(schema=summary,native_loop_reference=loop,
        numerical_sections_unchanged=True,launcher_Linux_syntax_passed=True,
        actual_prepared_admission_guard_rejected=True,
        actual_C6_placeholders_rejected=True,schema_fixture_not_actual_checkpoint6=True,
        actual_checkpoint6_inputs_verified=False,
        completion_guard=completion_guard_reference(launcher,plan))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--identity',required=True)
    parser.add_argument('--actual-inputs',action='store_true')
    args=parser.parse_args()
    assert re.fullmatch(r'cpumotussegment02(code|input)[0-9]{2}',args.identity)
    assert os.uname().sysname=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    assert not torch.cuda.is_initialized()
    code=ROOT.parents[1];source=os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==source
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    plan=json.loads((ROOT/'MOTUS-CURVE-GRPO-TRAIN02-CONTRACT.json').read_bytes())
    out=BASE/args.identity;out.mkdir()
    report=dict(status='checking',source_commit=source,GPU_seconds=0,
        model_or_simulator_constructed=False,GPU_training_admitted=False)
    write(out/'RESULT.json',report)
    try:
        assert plan['status']=='prepared'
        for name,expected in plan['source_files'].items():assert sha(code/name)==expected,name
        assert sha(ROOT/'motus_curve_entry.py')==plan['entry_sha256']
        assert sha(ROOT/'train_motus_curve02.sh')==plan['launcher_sha256']
        if args.actual_inputs:
            report.update(verify_inputs(plan),status='passed_actual_C6_prefix_inputs_for_6to10')
        else:
            assert not plan['budget_extension_approved']
            report.update(code_reference(plan),
                status='passed_segment6to10_launcher_schema_native_loop_CPU_reference')
        assert not torch.cuda.is_initialized()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:write(out/'RESULT.json',report)
    h=sha(out/'RESULT.json')
    print('SEGMENT02_CPU_QUALIFIED',str(out/'RESULT.json'),*[h[i:i+16] for i in range(0,64,16)],flush=True)


if __name__=='__main__':main()
