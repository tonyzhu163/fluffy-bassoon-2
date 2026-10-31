"""Bounded software mode of the existing CPU gate; ownership/proof code is reused."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

BASE='outputs/robotwin-sac65-cpu-compat-20261031-r6'
RUN='robotwin-ad-first-software-qual-20261031-r1'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stage_main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--sha',required=True)
    a=p.parse_args()
    if sha(a.stage)!=a.sha:raise ValueError('Stage context changed')
    c=json.loads(a.stage.read_text())
    module=load('qualified_local_stage',Path(__file__).with_name('stage_local_code.py'))
    module.stage(c['source'],c['pin'],c['output'],bootstrap=c['bootstrap'])

def main(started,gate):
    deadline=started+500
    p=argparse.ArgumentParser()
    for name in ('code-root','output','bootstrap-ticket','owned-helpers','software-profile'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--software-profile-sha256',required=True)
    a=p.parse_args();profile_raw=a.software_profile.read_bytes()
    gate.require(hashlib.sha256(profile_raw).hexdigest()==a.software_profile_sha256,'Software profile changed')
    profile=json.loads(profile_raw)
    gate.require(profile['run_id']==RUN and profile['global_work_seconds']==500
        and profile['scheduler_cap_seconds']==600 and profile['external_TERM_seconds']==570
        and profile['external_KILL_seconds']==580 and profile['bootstrap_wait_seconds']==120
        and profile['cleanup_reserve_seconds']==30 and profile['gpus']==0
        and profile['cpus']==8 and profile['memory_gib']==64,'Wrong software resources/clocks')
    hostname=socket.gethostname();code=a.code_root.resolve();out=a.output.resolve()
    gate.require(sys.platform=='linux' and out==Path('/mowenjie/fluffy-bassoon-2/outputs')/RUN
        and not out.exists() and not out.is_relative_to(code),'Need fresh allocated software output')
    out.mkdir();(out/'private').mkdir();(out/'tmp').mkdir()
    sys.stdout=sys.stderr=(out/'parent.log').open('x',buffering=1)
    gate.write(out/'intent.json',dict(run_id=RUN,auxiliary_pin=a.code_commit,profile_sha256=a.software_profile_sha256,
        hostname=hostname,profile=profile,physical_actions=0,SGD=0,full_corpus_projection=False,
        CPU_accounting='Charge actual scheduler duration once against proposed7200s phase total; no sealed credit'))
    def interrupted(signum,frame):raise RuntimeError('Owned software gate signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    runtime=None;failure=None;phases=[];results={};status='incomplete'
    try:
        gate.write(out/'bootstrap-intent.json',dict(ticket_path=str(a.bootstrap_ticket),max_wait_seconds=120,
            code_commit=a.code_commit,hostname=hostname,output=str(out),
            work_elapsed_seconds=time.monotonic()-started,work_deadline_seconds_from_start=500,
            bootstrap_expires_at_unix=time.time()+120,run_id=RUN))
        ref,receipt=gate.wait_bootstrap(a.bootstrap_ticket,code_commit=a.code_commit,output=out,
            hostname=hostname,deadline=deadline,run_id=RUN)
        gate.write(out/'bootstrap-receipt.json',receipt)
        raw=Path(ref['path']).read_bytes();gate.require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Resource bytes changed')
        r=json.loads(raw)
        gate.require(r['schema']=='robotwin_actual_cpu_allocation_v1' and r['hostname']==hostname
            and hostname==r['allocation_id']+'-master-0' and type(r['allocated_gpus']) is int
            and r['allocated_gpus']==0 and r['allocated_cpus']==8 and r['memory_gib']==64
            and r['observed_running_allocation'] is True,'Actual0GPU/8CPU/64GiB proof required')
        for key in ('proof','ax_proof'):gate.require(gate.sha(r[key]['path'])==r[key]['sha256'],'Actual UI proof changed')
        adapter=load('qualified_owned_cpu',code/BASE/'owned_cpu_runtime.py')
        runtime=adapter.CPUPhaseRuntime(a.owned_helpers,out/'private/ownership',deadline=deadline)
        observed,vis=gate.pre_mask_visibility(runtime,code,out,os.environ)
        gate.write(out/'allocation-before-mask.json',dict(resource=r,resource_sha256=ref['sha256'],
            environment_before_mask=observed['environment_before_mask'],visibility_worker=vis,
            actual_zero_gpu_attested=True,torch_imported=False))
        os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',TMPDIR=str(out/'tmp'))
        staging=load('qualified_stage',code/BASE/'stage_local_code.py')
        clean=staging.git_environment(os.environ)
        for key in tuple(os.environ):
            if key not in clean:os.environ.pop(key)
        def run(name,command,cwd,log,cap):
            available=deadline-time.monotonic()-30
            gate.require(available>5,'No software work time remains before cleanup')
            result=runtime.run(name,command,cwd,os.environ,log,min(cap,available))
            gate.require(result['cleanup']['clean'],'Owned cleanup failed: '+name)
            phases.append(dict(name=name,**result));gate.write(out/'phases.json',phases)
            return result
        roots={}
        for family in ('auxiliary','ad','teacher'):
            bound=profile[family] if family!='auxiliary' else dict(root=str(code),pin=a.code_commit,
                bootstrap=profile['auxiliary_bootstrap'])
            stage_out=out/('stage-'+family);stage_out.mkdir()
            c=dict(source=bound['root'],pin=bound['pin'],output=str(stage_out),bootstrap=bound['bootstrap'])
            path=stage_out/'context.json';gate.write(path,c)
            result=run('stage-'+family,[sys.executable,'-I','-B','-S',str(code/BASE/'software_profile.py'),
                '--stage',str(path),'--sha',sha(path)],code,stage_out/'console.log',30)
            gate.require(result['returncode']==0,'Exact staging failed: '+family)
            local=json.loads((stage_out/'local-code.json').read_text())
            gate.require(local['completed'] and local['full_clean_git_status']
                and local['original_head']==local['effective_head']==bound['pin'],'Wrong local source identity')
            roots[family]=Path(local['effective_code_root'])
        auxiliary=roots['auxiliary'];reader=out/'private/qualified-c62-reader.py'
        result=run('reader',['git','--no-pager','show',gate.READER_PIN+':'+gate.READER_PATH],auxiliary,reader,10)
        gate.require(result['returncode']==0 and sha(reader)==gate.READER_SHA,'Qualified corpus reader changed')
        def worker(family,phase):
            target=out/(family+'-'+phase);target.mkdir()
            bound=profile[family]
            c=dict(code_root=str(roots[family]),code_commit=bound['pin'],output=str(target),
                allocation_receipt=str(out/'allocation-before-mask.json'),
                allocation_receipt_sha256=sha(out/'allocation-before-mask.json'),
                site_packages=profile['site_packages'],shared_runtime_metadata=dict(
                    path=str(auxiliary/BASE/'shared-runtime-path-metadata.json'),
                    sha256=profile['shared_runtime_metadata_sha256']),chain_reader=str(reader),
                owned_helpers=str(a.owned_helpers),software_test_files=bound['tests'],
                deselected_nodes=bound.get('deselected_nodes',[]),expected_fixture_tests=bound['expected_tests'],
                teacher_plan=profile['teacher']['plan'],teacher_plan_sha256=profile['teacher']['plan_sha256'])
            path=target/'context.json';gate.write(path,c)
            result=run(family+'-'+phase,[sys.executable,'-I','-B','-S',str(auxiliary/BASE/'phase_worker.py'),
                '--phase',phase,'--context',str(path),'--context-sha256',sha(path)],roots[family],
                target/'console.log',120)
            ip=target/(phase+'-imports.json');gate.require(ip.is_file(),'Missing import/isolation receipt')
            imports=json.loads(ip.read_text())
            gate.require(not imports['denied_imports'] and not imports['forbidden_modules_loaded']
                and not imports['cuda_initialized'] and imports['optimizer_steps_permitted'] is False,
                'CPU isolation failed: '+family)
            passed=result['returncode']==0 and imports['error'] is None
            value=dict(passed=passed,code_commit=bound['pin'],imports_sha256=sha(ip),cleanup_clean=True,
                returncode=result['returncode'],scope='software only; no fit/teacher competence/robot admission')
            if passed and phase=='software-tests':
                counts=json.loads((target/'fixtures-summary.json').read_text())
                gate.require(counts==dict(tests=bound['expected_tests'],failures=0,errors=0,skipped=0),'Fixture coverage changed')
                value.update(counts=counts,JUnit_sha256=sha(target/'fixtures.junit.xml'),
                    deselected_legacy_nodes=bound.get('deselected_nodes',[]))
            gate.write(out/(family+'-'+phase+'-result.json'),value)
            return passed,value
        # AD is first and gets an independent durable receipt before teacher work.
        passed,results['ad']=worker('ad','software-tests')
        gate.require(passed,'AD software qualification failed; no automatic retry')
        passed,results['teacher_tests']=worker('teacher','software-tests')
        if passed:
            passed,results['teacher_source']=worker('teacher','source-admit')
        status='complete' if passed else 'AD_passed_teacher_incomplete'
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        cleanup_error=None
        if runtime is not None:
            try:runtime.close()
            except BaseException as exc:cleanup_error=str(exc)
        terminal=dict(run_id=RUN,status=status,failure=failure,cleanup_error=cleanup_error,
            parent_seconds=time.monotonic()-started,phases=phases,results=results,
            scheduler_cost='pending actual terminal UI receipt',physical_actions=0,VLA_queries=0,SGD=0)
        gate.write(out/'terminal.json',terminal)
        if cleanup_error:raise RuntimeError(cleanup_error)
    if status!='complete':raise RuntimeError('Teacher software incomplete; independent AD receipt retained')

if __name__=='__main__':stage_main()
