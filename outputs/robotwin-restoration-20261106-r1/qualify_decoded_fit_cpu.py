"""Actual-data, full-architecture CPU integration and fresh-process recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    code = ROOT.parents[1]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    output = BASE/'cpudecodedfit01'
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking',source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,
        full_planned_architecture=True,native_motus_or_simulator_constructed=False,
        test_tensor_loaded=False,learning_improvement_claim=False,children=[])

    def save():
        temporary = output/'RESULT.json.partial'
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output/'RESULT.json')

    def child(name,method,*,resume=None):
        command = [sys.executable,str(ROOT/'fit_decoded_student.py'),
            '--contract',str(ROOT/'DECODEDFITQUAL01-CONTRACT.json'),
            '--output',str(output/name),'--method',method,'--device','cpu']
        if resume:
            command += ['--resume',str(resume[0]),'--resume-sha256',resume[1]]
        with (output/(name+'.log')).open('x') as stream:
            run = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        path = output/name/'RESULT.json'
        value = json.loads(path.read_bytes()) if path.exists() else {}
        report['children'].append(dict(name=name,method=method,exit_code=run.returncode,
                                       status=value.get('status'),path=str(path)))
        save()
        assert run.returncode==0 and value['status']=='complete_offline_runtime_qualification',value
        assert value['full_planned_architecture'] and value['frozen_export_exact']
        assert not value['cuda_initialized'] and not value['test_tensor_loaded']
        return value

    save()
    try:
        reports = {}
        for method in ('ic_iql','dit','static_behavior_cloning'):
            report['active_child'] = method;save()
            reports[method] = child(method,method)
        first = next(row for row in reports['ic_iql']['checkpoints'] if row['update']==1)
        report['active_child'] = 'ic_iql-restore';save()
        restored = child('ic_iql-restore','ic_iql',resume=(output/'ic_iql'/first['file'],first['sha256']))
        assert restored['restored']['updates']==dict(value=1,actor=0)
        assert restored['optimizer_updates_this_process']==3
        assert restored['final_checkpoint_identity']==reports['ic_iql']['final_checkpoint_identity']
        assert restored['updates']==reports['ic_iql']['updates']==dict(value=2,actor=2)
        data_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        assert data_bytes <= 12*2**30
        report.update(status='passed',active_child=None,methods=list(reports),
            actual_data_updates_by_method={name:value['updates'] for name,value in reports.items()},
            fresh_process_restore_exact=True,full_learner_optimizer_sampler_rng_identity_exact=True,
            restored_value_to_actor_phase_transition=True,output_bytes=data_bytes,
            maximum_child_host_peak_bytes=max(value['host_peak_bytes'] for value in [*reports.values(),restored]),
            trained_policy_behavior_or_online_adaptation_qualified=False,complete_source_campaign=False)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        save()


if __name__ == '__main__':
    main()
