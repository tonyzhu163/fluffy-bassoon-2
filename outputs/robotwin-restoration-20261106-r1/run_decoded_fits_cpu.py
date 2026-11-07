"""Run the three bounded offline pilot controls on allocated CPU resources."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert sys.platform=='linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    code = ROOT.parents[1]
    source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    assert source==os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    plan = json.loads((ROOT/'DECODEDFIT01-CONTRACT.json').read_bytes())
    assert plan['intent']=='student_fitting'
    output = BASE/'decodedfit01'
    output.mkdir()
    started = time.monotonic()
    cores = len(os.sched_getaffinity(0))
    cpu_limit = Path('/sys/fs/cgroup/cpu.max')
    if cpu_limit.exists():
        quota,period = cpu_limit.read_text().split()
        if quota!='max':
            cores = min(cores,max(1,int(quota)//int(period)))
    parallel = min(3,max(1,cores//4))
    memory_max = Path('/sys/fs/cgroup/memory.max')
    if memory_max.exists() and memory_max.read_text().strip()!='max':
        limit = int(memory_max.read_text());current = int(memory_max.with_name('memory.current').read_text())
        available = int(.9*limit)-current
        assert available>=4*2**30,'Insufficient allocated CPU fitting memory'
        parallel = min(parallel,max(1,available//(4*2**30)))
    waiting = list(plan['methods'])
    running,streams = {},[]
    report = dict(status='fitting',source_commit=source,gpu_seconds=0,
        maximum_concurrent_fits=parallel,allocated_cpu_limit=cores,children={},
        test_tensor_loaded=False,full_source_campaign_admitted=False,
        learning_improvement_claim=False,online_deployment_qualified=False)

    def save():
        temporary = output/'RESULT.json.partial'
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output/'RESULT.json')

    save()
    try:
        while waiting or running:
            while waiting and len(running)<parallel:
                name = waiting.pop(0)
                command = [sys.executable,str(ROOT/'fit_decoded_student.py'),
                    '--contract',str(ROOT/'DECODEDFIT01-CONTRACT.json'),
                    '--output',str(output/name),'--method',name,'--device','cpu']
                stream = (output/(name+'.log')).open('x');streams.append(stream)
                process = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                running[name] = process
                report['children'][name] = dict(pid=process.pid,status='running')
                save()
            for name,process in list(running.items()):
                result = process.poll()
                if result is None and time.monotonic()-started>plan['maximum_cpu_seconds']:
                    process.send_signal(__import__('signal').SIGINT)
                    try:
                        result = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill();result=process.wait()
                if result is not None:
                    path = output/name/'RESULT.json'
                    value = json.loads(path.read_bytes()) if path.exists() else {}
                    report['children'][name].update(exit_code=result,status=value.get('status','missing_result'),
                        updates=value.get('updates'),path=str(path))
                    del running[name]
                    save()
            if waiting or running:
                time.sleep(10)
        success = all(row['exit_code']==0 and row['status']=='complete_offline_student_fitting'
                      for row in report['children'].values())
        report['status'] = 'completed_offline_fits' if success else 'failed'
        report['output_bytes'] = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        assert report['output_bytes']<=plan['maximum_new_output_gib']*2**30
        if not success:
            raise RuntimeError('Inspect failed child before any fresh output/retry')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        save()
        for stream in streams:
            stream.close()


if __name__=='__main__':
    main()
