"""Dedicated0GPU production-glue tests; tiny CPU fixtures are declared honestly."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import time
import xml.etree.ElementTree as ET

from fluffy_bassoon.meta_rl.vla_source40_cpu_qualification import (
    CPU_ENV,cgroup_resources,require_zero_gpu,sha,write)

FORMAT='source40_ppo_production_zero_gpu_test_v1'
TESTS=['tests/test_vla_source40_batch14_production.py',
    'tests/test_vla_source40_batch14_production_shared.py']


def finish(output,code):
    path=Path(output)/'cpu-worker-clock.json';clock=json.loads(path.read_text())
    ended=time.time()
    clock.setdefault('closure_history',[]).append(dict(ended_unix=ended,exit_code=int(code)))
    clock.update(status='finished',ended_unix=ended,exit_code=int(code),
        charged_cpu_worker_seconds=math.ceil(max(0.,ended-clock['started_unix']))+60,
        charged_gpu_seconds=0 if clock.get('zero_gpu_attested') else None)
    clock['cpu_time_overrun']=clock['charged_cpu_worker_seconds']>1200
    write(path,clock)


def run(a):
    code=Path(__file__).resolve().parents[2]
    if (sys.platform!='linux' or socket.gethostname()!=a.expected_hostname
            or not socket.gethostname().startswith('job-') or sys.version_info[:2]!=(3,12)
            or Path(sys.prefix).resolve()!=CPU_ENV.resolve()
            or any(bad in os.environ.get('PYTHONPATH','') for bad in ('cu128-overlay','python3.11','cp311'))
            or a.output.name!=a.run or not a.run or any(not(c.isalnum() or c=='-') for c in a.run)
            or not math.isfinite(a.started_unix) or not 0<=time.time()-a.started_unix<1100):
        raise ValueError('Require exact identified Baige CPU job, Python3.12 base, fresh identity and20min cap')
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()!=a.source
            or subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()):
        raise ValueError('Require exact clean reviewed CPU test source')
    a.output.mkdir(parents=True,exist_ok=False)
    clock=dict(format=FORMAT,source_commit=a.source,run_id=a.run,started_unix=a.started_unix,
        status='started',maximum_cpu_worker_seconds=1200,execution_seconds=1100,
        startup_allowance_seconds=60,zero_gpu_attested=False,charged_gpu_seconds=None,
        interrupted_charge_cpu_seconds=1200)
    write(a.output/'cpu-worker-clock.json',clock)
    contract=dict(format=FORMAT,source_commit=a.source,run_id=a.run,hostname=a.expected_hostname,
        tests=TESTS,requested=dict(gpu_count=0,cpu_cores=4,memory_bytes=16*1024**3,wall_seconds=1200),
        tiny_cpu_model_test_fixtures_may_construct=True,cpu_optimizer_test_fixtures_may_construct=True,
        vla_model_construction=False,policy_forward=False,backward=False,optimizer_step_count=0,
        simulator_actions=0,production_run_created=False,shared_pfs_integration_required=True,
        runtime=sys.executable,python_abi=sysconfig.get_config_var('SOABI'))
    write(a.output/'contract.json',contract)
    rc=1
    try:
        import torch
        attestation=require_zero_gpu(torch)
        resources=cgroup_resources()
        attestation.update(hostname=a.expected_hostname,python=sys.executable,sys_prefix=sys.prefix,
            python_abi=sysconfig.get_config_var('SOABI'),resources=resources)
        write(a.output/'zero-gpu-attestation.json',attestation)
        clock.update(zero_gpu_attested=True,charged_gpu_seconds=0);write(a.output/'cpu-worker-clock.json',clock)
        if resources['effective_cpu_quota_cores']!=4 or resources['memory_limit_bytes']!=16*1024**3:
            raise ValueError('Measured CPU cgroup does not match4CPU/16GiB request')
        env=dict(os.environ,FLUFFY_REQUIRE_PPO_SHARED_CHECKPOINT='1',
            FLUFFY_PPO_CPU_SOURCE=a.source,FLUFFY_PPO_CPU_RECEIPT_DIR=str(a.output))
        command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            '--basetemp='+str(a.output/'pytest-tmp'),'--junitxml='+str(a.output/'junit.xml'),*TESTS]
        remaining=a.started_unix+1060-time.time()
        if remaining<=0:raise TimeoutError('Dedicated CPU worker deadline reached before tests')
        write(a.output/'test-intent.json',dict(command=command,timeout_seconds=remaining,started_unix=time.time()))
        with (a.output/'tests.log').open('x') as log:
            completed=subprocess.run(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=remaining,check=False)
        if completed.returncode:raise RuntimeError('Focused production tests failed; original logs/artifacts retained')
        junit=ET.parse(a.output/'junit.xml').getroot()
        suites=[junit] if junit.tag=='testsuite' else list(junit.findall('testsuite'))
        summary={key:sum(int(row.attrib.get(key,0)) for row in suites) for key in ('tests','failures','errors','skipped')}
        shared=json.loads((a.output/'shared-checkpoint-integration.json').read_text())
        if (summary['tests']<1 or any(summary[k] for k in ('failures','errors','skipped'))
                or shared.get('passed') is not True or torch.cuda.is_initialized() or torch.cuda.device_count()!=0):
            raise RuntimeError('Require all tests/actual PFS integration without skips and CUDA never initialized')
        write(a.output/'index.json',dict(contract,completed=True,test_summary=summary,
            zero_gpu_attestation_sha256=sha(a.output/'zero-gpu-attestation.json'),
            junit_sha256=sha(a.output/'junit.xml'),shared_checkpoint_integration=shared,
            shared_checkpoint_receipt_sha256=sha(a.output/'shared-checkpoint-integration.json'),
            allocated_gpu_count=0,cuda_uninitialized_at_completion=True,gpu_seconds=0,
            interpretation='CPU test fixtures and actual lossless shared-checkpoint migration only; no production launch/SGD/performance result'))
        rc=0
    except BaseException as error:
        write(a.output/'failure.json',dict(completed=False,error=repr(error),partial_artifacts_preserved=True,
            gpu_seconds=0 if clock['zero_gpu_attested'] else None,production_run_created=False))
        raise
    finally:finish(a.output,rc)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation',choices=('run','finish'));ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source');ap.add_argument('--run');ap.add_argument('--expected-hostname')
    ap.add_argument('--started-unix',type=float);ap.add_argument('--exit-code',type=int)
    a=ap.parse_args()
    if a.operation=='finish':finish(a.output,a.exit_code)
    else:run(a)


if __name__=='__main__':main()
