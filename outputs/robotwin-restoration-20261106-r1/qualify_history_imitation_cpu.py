"""Qualify full-history imitation and exact full-model actor-stage recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fit_decoded_student import BASE, ROOT, write


def main():
    assert sys.platform=='linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    code=ROOT.parents[1]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    out=BASE/'cpuhistoryimitation01';out.mkdir()
    report=dict(status='checking',source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,
        test_tensor_loaded=False,chronological_ad_qualified=False,robot_adaptation_qualified=False)
    started=time.monotonic()
    write(out/'RESULT.json',report)
    try:
        files=['tests/test_chunk_training.py','tests/test_chunk_value.py','tests/test_chunk_controller.py',
               'tests/test_robotwin_batch.py']
        with (out/'core-tests.log').open('x') as stream:
            run=subprocess.run([sys.executable,'-m','pytest','-q',*files],cwd=code,
                stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        assert run.returncode==0
        report['core_tests']=dict(files=files,exit_code=run.returncode)
        write(out/'RESULT.json',report)
        contract=ROOT/'HISTORYIMITATION-QUAL01-CONTRACT.json'

        def fit(name,resume=None):
            command=[sys.executable,str(ROOT/'fit_decoded_student.py'),'--contract',str(contract),
                '--output',str(out/name),'--method','history_behavior_cloning','--device','cpu']
            if resume:command+=['--resume',str(resume[0]),'--resume-sha256',resume[1]]
            with (out/(name+'.log')).open('x') as stream:
                run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            assert run.returncode==0
            value=json.loads((out/name/'RESULT.json').read_bytes())
            assert value['status']=='complete_offline_runtime_qualification'
            assert value['full_planned_architecture'] and value['frozen_export_exact']
            assert not value['cuda_initialized'] and not value['test_tensor_loaded']
            assert value['contract']['fitting']['method']=='history_behavior_cloning'
            assert value['contract']['fitting']['mode']=='full' and value['contract']['objective']['objective']=='ad'
            assert not value['contract']['data']['chronological_learning_histories']
            return value

        original=fit('original')
        first=next(row for row in original['checkpoints'] if row['update']==1)
        restored=fit('restored',(out/'original'/first['file'],first['sha256']))
        assert restored['restored']['updates']==dict(value=0,actor=1)
        assert restored['optimizer_updates_this_process']==1
        assert original['updates']==restored['updates']==dict(value=0,actor=2)
        assert original['final_checkpoint_identity']==restored['final_checkpoint_identity']
        report.update(status='passed',actual_data_full_model_updates=original['updates'],
            strict_frozen_export=True,full_learner_optimizer_sampler_rng_recovery_exact=True,
            frozen_actor_export_tensors_exact_via_identical_learners=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['seconds']=time.monotonic()-started;write(out/'RESULT.json',report)


if __name__=='__main__':main()
