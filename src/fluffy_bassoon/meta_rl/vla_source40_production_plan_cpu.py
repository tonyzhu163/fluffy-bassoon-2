"""Prepare the source production plan in an allocated0GPU CPU container.

Metadata only: no tests, model construction, forward pass, optimizer, rollouts,
funding, source-ledger mutation or corpus copying. Existing parent_record owns
terminal parent validation and full-recovery hashes.
"""
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

from .vla_source40_cpu_qualification import CPU_ENV, cgroup_resources, require_zero_gpu, sha, write


FORMAT='source40_unfunded_production_plan_cpu_v1'
PROJECT=Path('/mowenjie/fluffy-bassoon-2')
BRIDGE_SOURCE='cf49bfac65a98cc7c4cef7fb35df4a6a18e20c0b'
BRIDGES={key:str(PROJECT/'outputs'/('libero40-sac-source-bridge-20261031-r1-'+name+'-train'))
    for key,name in [('libero_spatial:00/train','s00'),('libero_object:00/train','o00')]}


def finish(root,exit_code):
    path=Path(root)/'cpu-worker-clock.json';value=json.loads(path.read_text());ended=time.time()
    value.setdefault('closure_history',[]).append(dict(ended_unix=ended,exit_code=int(exit_code)))
    value.update(status='finished',ended_unix=ended,exit_code=int(exit_code),
        charged_cpu_worker_seconds=math.ceil(max(0.,ended-value['started_unix']))+60,
        charged_gpu_seconds=0 if value.get('zero_gpu_attested') else None)
    value['cpu_time_overrun']=value['charged_cpu_worker_seconds']>1200
    write(path,value)
    return value


def run(args):
    code=Path(__file__).resolve().parents[3]
    if (sys.platform!='linux' or socket.gethostname()!=args.expected_hostname
            or not socket.gethostname().startswith('job-') or sys.version_info[:2]!=(3,12)
            or Path(sys.prefix).resolve()!=CPU_ENV.resolve()
            or any(bad in os.environ.get('PYTHONPATH','') for bad in ('cu128-overlay','python3.11','cp311'))
            or not math.isfinite(args.started_unix) or not 0<=time.time()-args.started_unix<1140):
        raise ValueError('Require identified allocated Python3.12 CPU runtime and original20min deadline')
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()!=args.source
            or subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
            or args.output.name!=args.run or not args.run or not args.production_prefix
            or any(not(c.isalnum() or c=='-') for c in args.run+args.production_prefix)):
        raise ValueError('Require exact clean source, fresh CPU identity and plain production prefix')
    args.output.mkdir(parents=True,exist_ok=False)
    clock=dict(format=FORMAT,source_commit=args.source,run_id=args.run,hostname=args.expected_hostname,
        started_unix=args.started_unix,status='started',maximum_cpu_worker_seconds=1200,
        execution_seconds=1140,startup_allowance_seconds=60,zero_gpu_attested=False,
        charged_gpu_seconds=None,interrupted_charge_cpu_seconds=1200)
    write(args.output/'cpu-worker-clock.json',clock)
    result=1
    try:
        import torch
        attestation=require_zero_gpu(torch)
        attestation.update(hostname=args.expected_hostname,python=sys.executable,sys_prefix=sys.prefix,
            python_abi=sysconfig.get_config_var('SOABI'),resources=cgroup_resources())
        write(args.output/'zero-gpu-attestation.json',attestation)
        if any(value is not True for value in attestation['resources']['cgroup_matches_request'].values()):
            raise ValueError('Actual CPU cgroup does not match0GPU/4CPU/16GiB request')
        clock.update(zero_gpu_attested=True,charged_gpu_seconds=0);write(args.output/'cpu-worker-clock.json',clock)
        from .vla_source40_topup import make_plan
        from .vla_source40_topup_worker import assigned, immutable, train_admission
        receipts=code/'outputs/libero40-bulk-approved-20261030-r1/terminal-source-receipts.json'
        immutable(args.output/'contract.json',dict(format=FORMAT,source_commit=args.source,
            run_id=args.run,production_prefix=args.production_prefix,receipts=str(receipts),receipts_sha256=sha(receipts),
            evidence_root=str(code/'outputs'),overrides=BRIDGES,tests_run=False,model_construction=False,
            inference=False,optimizer_updates=0,simulator_actions=0,source_ledger_writes=False,budget_authority=False))
        immutable(args.output/'overrides.json',BRIDGES)
        immutable(args.output/'plan-intent.json',dict(started_unix=time.time(),operation='existing make_plan production',
            note='Hash existing terminal recoveries/metadata only; no parent modification or raw corpus copying'))
        plan=make_plan(receipts,project_root=PROJECT,evidence_root=code/'outputs',
            source_commit=args.source,mode='production',overrides=BRIDGES)
        admission=train_admission(plan)
        retained={row['slot']:row for row in admission['retained_only_source_leaves']}
        if (plan['maximum_charged_including_overhead']!=998554 or plan['inherited_action_charge']!=457851
                or admission['all40_admitted_train_count']!=40 or admission['launchable_train_count']!=38
                or set(retained)!=set(BRIDGES)
                or any(row['parent']['source_commit']!=BRIDGE_SOURCE for row in retained.values())):
            raise ValueError('Production plan differs from exact all40/38-launchable/two retained bridge/source-cost design')
        immutable(args.output/'production.json',plan)
        workers=[dict(worker=worker,slots=assigned(plan,args.production_prefix,worker,PROJECT/'outputs')) for worker in range(8)]
        slots=[row['slot'] for worker in workers for row in worker['slots']]
        if len(slots)!=38 or len(set(slots))!=38 or set(slots)&set(BRIDGES):
            raise ValueError('All38 production assignments must be unique and exclude quota-filled bridges')
        immutable(args.output/'assignments.json',dict(source_admission=admission,workers=workers,
            all40_coverage='All40 admitted;38 new assignments and2 retained terminal leaves; not completed collection'))
        template=json.loads((code/'outputs/libero40-sac-production-worker-budget-20261031-r1/budget-template.json').read_text())
        template.update(funding_status='UNFUNDED_TEMPLATE',source_commit=args.source,plan_sha256=plan['plan_sha256'],
            campaign_prefix=args.production_prefix,campaign_root=str(PROJECT/'outputs'/args.production_prefix))
        immutable(args.output/'unfunded-macro-budget.json',template)
        if torch.cuda.is_initialized() or torch.cuda.device_count()!=0:
            raise RuntimeError('CUDA state changed during metadata preparation')
        immutable(args.output/'index.json',dict(format=FORMAT,completed=True,source_commit=args.source,
            production_plan_path=str((args.output/'production.json').resolve()),production_plan_file_sha256=sha(args.output/'production.json'),
            production_plan_content_sha256=plan['plan_sha256'],assignments_sha256=sha(args.output/'assignments.json'),
            zero_gpu_attestation_sha256=sha(args.output/'zero-gpu-attestation.json'),
            unfunded_macro_budget_sha256=sha(args.output/'unfunded-macro-budget.json'),
            source_admission=admission,assigned_streams=38,inherited_source_resource_actions=457851,
            maximum_cumulative_source_actions=998554,allocated_gpu_count=0,cuda_initialized=False,
            tests_run=False,model_construction=False,simulator_actions=0,source_ledger_writes=False,
            budget_authority=False,interpretation='Production metadata prepared, not a resource grant, collection or scientific result'))
        result=0
        return 0
    except BaseException as error:
        write(args.output/'failure.json',dict(completed=False,error=repr(error),partial_outputs_preserved=True,
            simulator_actions=0,source_ledger_writes=False,budget_authority=False))
        raise
    finally:finish(args.output,result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    prep=sub.add_parser('run')
    for name in ('source','run','production-prefix','expected-hostname'):prep.add_argument('--'+name,required=True)
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--started-unix',type=float,required=True)
    close=sub.add_parser('finish');close.add_argument('--output',type=Path,required=True);close.add_argument('--exit-code',type=int,required=True)
    args=parser.parse_args(argv)
    if args.operation=='finish':print(json.dumps(finish(args.output,args.exit_code)));return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
