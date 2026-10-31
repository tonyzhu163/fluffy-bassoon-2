"""Closed CPU profiles of the existing gate; ownership/proof code is reused."""
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
FINAL_RUN='robotwin-final100-ad-export-20261031-r1'
RESIDUAL_RUN='robotwin-final100-ad-export-20261031-r2'
PROOF_RECOVERY_RUN='robotwin-final100-ad-export-20261031-r3'
QUALIFIED_CLOSEOUT_RUN='robotwin-final100-ad-export-20261031-r4'

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
    p=argparse.ArgumentParser()
    for name in ('code-root','output','bootstrap-ticket','owned-helpers','software-profile'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--software-profile-sha256',required=True)
    a=p.parse_args();profile_raw=a.software_profile.read_bytes()
    gate.require(hashlib.sha256(profile_raw).hexdigest()==a.software_profile_sha256,'Software profile changed')
    profile=json.loads(profile_raw)
    run=profile['run_id']
    qualified_closeout=run==QUALIFIED_CLOSEOUT_RUN
    proof_recovery=run in (PROOF_RECOVERY_RUN,QUALIFIED_CLOSEOUT_RUN)
    residual=run in (RESIDUAL_RUN,PROOF_RECOVERY_RUN,QUALIFIED_CLOSEOUT_RUN)
    finalbank=run in (FINAL_RUN,RESIDUAL_RUN,PROOF_RECOVERY_RUN,QUALIFIED_CLOSEOUT_RUN)
    clocks=(run,profile['global_work_seconds'],profile['scheduler_cap_seconds'],profile['external_TERM_seconds'],profile['external_KILL_seconds'])
    gate.require(clocks in ((RUN,500,600,570,580),('robotwin-ad-first-software-qual-20261031-r2',380,480,450,460),
        ('robotwin-ad500-software-qual-20261031-r1',250,300,270,280),
        (FINAL_RUN,6720,7020,6790,6800),(RESIDUAL_RUN,5700,6000,5770,5780),
        (PROOF_RECOVERY_RUN,5520,5820,5590,5600),(QUALIFIED_CLOSEOUT_RUN,5400,5700,5470,5480))
        and profile['bootstrap_wait_seconds']==120
        and profile['cleanup_reserve_seconds']==30 and profile['gpus']==0
        and profile['cpus']==8 and profile['memory_gib']==64,'Wrong software resources/clocks')
    if finalbank:
        gate.require(profile['kind']=='robotwin_final100_ad_cpu_v1'
            and (profile['ad']['pin']=='7401cd50c038a4cb35149eceeb72128e9a368cc3' if not residual else
                 profile['ad']['pin']==profile['compatibility_source_pin']
                 and isinstance(profile['ad']['pin'],str) and len(profile['ad']['pin'])==40)
            and profile['prior_software_actual_seconds']==170
            and profile['old_sealed_credit_seconds']==0
            and 170+profile.get('prior_bank_actual_seconds',0)+profile.get('prior_bootstrap_actual_seconds',0)+profile.get('prior_schema_actual_seconds',0)+profile['scheduler_cap_seconds']+profile['unrequested_seconds']==7200
            and profile['unrequested_seconds']==(202 if qualified_closeout else 171 if proof_recovery else 141 if residual else 10),'Wrong final100 conditional residual')
        if residual:
            ref=profile['prior_bank_terminal_accounting']
            gate.require(sha(ref['path'])==ref['sha256'],'Prior bank terminal accounting changed')
            previous=json.loads(Path(ref['path']).read_text())
            gate.require(previous['job']=='job-twcl47ph8lnz' and previous['scheduler_state']=='Failed'
                and previous['scheduler_running_seconds']==profile['prior_bank_actual_seconds']==889
                and previous['GPUs']==0 and previous['remaining_approved_CPU_worker_seconds']==6141,
                'Residual must charge actual889 once; never renew7020 or borrow seals')
        if proof_recovery:
            ref=profile['prior_bootstrap_terminal_accounting']
            gate.require(sha(ref['path'])==ref['sha256'],'Prior bootstrap accounting changed')
            previous=json.loads(Path(ref['path']).read_text())
            gate.require(previous['job']=='job-jbvfhkq3gfhd' and previous['scheduler_state']=='Failed'
                and previous['scheduler_running_seconds']==profile['prior_bootstrap_actual_seconds']==150
                and previous['GPUs']==0 and previous['tests_run']==0 and previous['data_phases_run']==0
                and previous['remaining_approved_CPU_worker_seconds']==5991,'Charge bootstrap150 once; no clock reset')
        if qualified_closeout:
            ref=profile['prior_schema_terminal_accounting']
            gate.require(sha(ref['path'])==ref['sha256'],'Prior schema accounting changed')
            previous=json.loads(Path(ref['path']).read_text())
            gate.require(previous['job']=='job-y8ap2axem2yk' and previous['scheduler_state']=='Failed'
                and previous['scheduler_running_seconds']==profile['prior_schema_actual_seconds']==89
                and previous['GPUs']==0 and previous['tests_passed']==146
                and previous['remaining_approved_CPU_worker_seconds']==5902,'Charge schema89 once; no new allowance')
        ref=profile['prior_cpu_closure']
        gate.require(sha(a.code_root/ref['path'])==ref['sha256'],'Software closure changed')
        closed=json.loads((a.code_root/ref['path']).read_text())
        gate.require(closed['aggregate_software_actual_worker_seconds']==170
            and closed['GPU_seconds']==0,'Charge actual software170 exactly once')
        checked={}
        for name,ref in profile['qualified_software'].items():
            gate.require(sha(a.code_root/ref['path'])==ref['sha256'],'Qualified component receipt changed: '+name)
            checked[name]=json.loads((a.code_root/ref['path']).read_text())
        gate.require(checked['assembly']['status']=='passed_closed_SAC65_CPU_compatibility_only'
            and checked['assembly']['failure'] is None and checked['ad']['passed'] is True
            and checked['ad']['code_commit']=='7401cd50c038a4cb35149eceeb72128e9a368cc3'
            and checked['ad']['counts']==dict(tests=108,failures=0,errors=0,skipped=0),
            'Existing assembly/AD software qualification required, not another test run')
        ref=profile['human_phase_authority']
        gate.require(isinstance(ref,dict),'Human phase funding is PENDING; prepared profile cannot launch')
        gate.require(sha(ref['path'])==ref['sha256'],'Human authority receipt changed')
        authority=json.loads(Path(ref['path']).read_text())
        gate.require(authority['schema']=='robotwin_ad_first_human_phase_authority_v1'
            and authority['direct_human_authorization'] is True
            and authority['CPU_worker_seconds']==7200
            and authority['new_storage_bytes']==128<<30
            and authority['human_message_reference'], 'Require verified direct human phase authority')
    if 'readout' in profile:
        prior=profile['prior_cpu_closure']
        gate.require(sha(a.code_root/prior['path'])==prior['sha256'],'Closed software cost receipt changed')
        closed=json.loads((a.code_root/prior['path']).read_text())
        gate.require(run=='robotwin-ad500-software-qual-20261031-r1'
            and closed['status']=='CLOSED_NO_ADMITTED_WORK' and closed['actual_worker_seconds']==139
            and closed['unused_sealed_seconds']==461 and closed['old_sealed_borrow']==0
            and profile['scheduler_cap_seconds']==300 and 139+300<=600
            and profile['old_sealed_credit_seconds']==0,
            'New readout-only CPU bound must count139 once, never borrow closed461 or add a new600')
    deadline=started+profile['global_work_seconds']
    hostname=socket.gethostname();code=a.code_root.resolve();out=a.output.resolve()
    gate.require(sys.platform=='linux' and out==Path('/mowenjie/fluffy-bassoon-2/outputs')/run
        and not out.exists() and not out.is_relative_to(code),'Need fresh allocated software output')
    out.mkdir();(out/'private').mkdir();(out/'tmp').mkdir()
    sys.stdout=sys.stderr=(out/'parent.log').open('x',buffering=1)
    gate.write(out/'intent.json',dict(run_id=run,auxiliary_pin=a.code_commit,profile_sha256=a.software_profile_sha256,
        hostname=hostname,profile=profile,physical_actions=0,SGD=0,full_corpus_projection=finalbank,
        CPU_accounting='Charge actual scheduler duration once against proposed7200s phase total; no sealed credit'))
    def interrupted(signum,frame):raise RuntimeError('Owned software gate signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    runtime=None;failure=None;phases=[];results={};status='incomplete'
    try:
        gate.write(out/'bootstrap-intent.json',dict(ticket_path=str(a.bootstrap_ticket),max_wait_seconds=120,
            code_commit=a.code_commit,hostname=hostname,output=str(out),
            work_elapsed_seconds=time.monotonic()-started,work_deadline_seconds_from_start=profile['global_work_seconds'],
            bootstrap_expires_at_unix=time.time()+120,run_id=run))
        ref,receipt=gate.wait_bootstrap(a.bootstrap_ticket,code_commit=a.code_commit,output=out,
            hostname=hostname,deadline=deadline,run_id=run)
        gate.write(out/'bootstrap-receipt.json',receipt)
        raw=Path(ref['path']).read_bytes();gate.require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Resource bytes changed')
        r=json.loads(raw)
        gate.require(r['schema']=='robotwin_actual_cpu_allocation_v1' and r['hostname']==hostname
            and hostname==r['allocation_id']+'-master-0' and type(r['allocated_gpus']) is int
            and r['allocated_gpus']==0 and r['allocated_cpus']==8 and r['memory_gib']==64
            and r['observed_running_allocation'] is True,'Actual0GPU/8CPU/64GiB proof required')
        for key in ('proof','ax_proof'):gate.require(gate.sha(r[key]['path'])==r[key]['sha256'],'Actual UI proof changed')
        adapter=load('qualified_owned_cpu',code/BASE/'owned_cpu_runtime.py')
        runtime=adapter.CPUPhaseRuntime(a.owned_helpers,out/'private/ownership',deadline=deadline,
            clock_profile='final100-ad-v1' if finalbank else 'compatibility')
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
        def run_phase(name,command,cwd,log,cap):
            available=deadline-time.monotonic()-30
            gate.require(available>5,'No software work time remains before cleanup')
            result=runtime.run(name,command,cwd,os.environ,log,min(cap,available))
            gate.require(result['cleanup']['clean'],'Owned cleanup failed: '+name)
            phases.append(dict(name=name,**result));gate.write(out/'phases.json',phases)
            return result
        roots={}
        def stage_family(family):
            bound=profile[family] if family!='auxiliary' else dict(root=str(code),pin=a.code_commit,
                bootstrap=profile['auxiliary_bootstrap'])
            stage_out=out/('stage-'+family);stage_out.mkdir()
            c=dict(source=bound['root'],pin=bound['pin'],output=str(stage_out),bootstrap=bound['bootstrap'])
            path=stage_out/'context.json';gate.write(path,c)
            result=run_phase('stage-'+family,[sys.executable,'-I','-B','-S',str(code/BASE/'software_profile.py'),
                '--stage',str(path),'--sha',sha(path)],code,stage_out/'console.log',30)
            gate.require(result['returncode']==0,'Exact staging failed: '+family)
            local=json.loads((stage_out/'local-code.json').read_text())
            gate.require(local['completed'] and local['full_clean_git_status']
                and local['original_head']==local['effective_head']==bound['pin'],'Wrong local source identity')
            expected=profile['auxiliary_file_sha256'] if family=='auxiliary' and finalbank else bound.get('file_sha256')
            if expected is not None:
                gate.require(local['bootstrap_sha256']==expected,'Prepared file identities changed: '+family)
            roots[family]=Path(local['effective_code_root'])
        for family in (('auxiliary','readout') if 'readout' in profile else
                       ('auxiliary',) if 'ad_predecessor' in profile else ('auxiliary','ad')):stage_family(family)
        auxiliary=roots['auxiliary'];reader=out/'private/qualified-c62-reader.py'
        result=run_phase('reader',['git','--no-pager','show',gate.READER_PIN+':'+gate.READER_PATH],auxiliary,reader,10)
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
                deselected_nodes=bound.get('deselected_nodes',[]),expected_fixture_tests=bound['expected_tests'])
            if 'teacher' in profile:
                c.update(teacher_plan=profile['teacher']['plan'],teacher_plan_sha256=profile['teacher']['plan_sha256'])
            path=target/'context.json';gate.write(path,c)
            result=run_phase(family+'-'+phase,[sys.executable,'-I','-B','-S',str(auxiliary/BASE/'phase_worker.py'),
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
        if finalbank:
            (out/'private/exports').mkdir()
            if qualified_closeout:
                refs=profile['qualified_schema_predecessor'];prior={}
                for name,ref in refs.items():
                    gate.require(sha(ref['path'])==ref['sha256'],'Qualified schema receipt changed: '+name)
                    prior[name]=json.loads(Path(ref['path']).read_text()) if name!='junit' else None
                value=prior['result'];imports=prior['imports'];terminal=prior['terminal'];cleanup=prior['cleanup']
                gate.require(value['passed'] and value['returncode']==0 and value['cleanup_clean']
                    and value['code_commit']==profile['ad']['pin']=='e5675998dd130ff722c922f1cf681f0a96f44eff'
                    and value['counts']==dict(tests=146,failures=0,errors=0,skipped=0)
                    and not value['deselected_legacy_nodes']
                    and value['JUnit_sha256']==refs['junit']['sha256']
                    and value['imports_sha256']==refs['imports']['sha256']
                    and imports['error'] is None and not imports['denied_imports']
                    and not imports['forbidden_modules_loaded'] and not imports['cuda_initialized']
                    and imports['optimizer_steps_permitted'] is False
                    and terminal['run_id']==PROOF_RECOVERY_RUN and terminal['status']=='incomplete'
                    and terminal['failure']=='ValueError: Finalbank phase failed; preserve output, no automatic retry: reuse'
                    and terminal['results']['schema_regressions']==value and terminal['cleanup_error'] is None
                    and terminal['physical_actions']==terminal['VLA_queries']==terminal['SGD']==0
                    and cleanup['clean'] and not cleanup['sessions'] and not cleanup['survivors']
                    and cleanup['adopted']['clean'],'Qualified schema evidence does not match actual closed R3')
                results['schema_regressions']=dict(value,adopted_from=refs,new_tests_run=0)
                gate.write(out/'ad-software-tests-result.json',results['schema_regressions'])
            elif residual:
                passed,results['schema_regressions']=worker('ad','software-tests')
                gate.require(passed,'Necessary dual-encoding regressions failed; preserve output')
            stages=(('reuse',180),('train',2400),('validation',600),('admission',400)) if residual else (
                ('inventory',180),('projection',2800),('train',2400),('validation',600),('admission',400))
            for phase,cap in stages:
                target=out/('finalbank-'+phase);target.mkdir()
                c=dict(code_root=str(roots['ad']),code_commit=profile['ad']['pin'],output=str(target),
                    finalbank_root=str(out),finalbank=profile['finalbank'],
                    compatibility_source_pin=profile['ad']['pin'],
                    phase_deadline=min(deadline-30,time.monotonic()+cap),
                    allocation_receipt=str(out/'allocation-before-mask.json'),
                    allocation_receipt_sha256=sha(out/'allocation-before-mask.json'),
                    site_packages=profile['site_packages'],shared_runtime_metadata=dict(
                        path=str(auxiliary/BASE/'shared-runtime-path-metadata.json'),
                        sha256=profile['shared_runtime_metadata_sha256']),chain_reader=str(reader))
                path=target/'context.json';gate.write(path,c)
                name='finalbank-'+phase
                result=run_phase(name,[sys.executable,'-I','-B','-S',str(auxiliary/BASE/'phase_worker.py'),
                    '--phase',name,'--context',str(path),'--context-sha256',sha(path)],roots['ad'],
                    target/'console.log',cap)
                ip=target/(name+'-imports.json')
                gate.require(ip.is_file(),'Missing finalbank isolation receipt')
                imports=json.loads(ip.read_text())
                gate.require(result['returncode']==0 and imports['error'] is None
                    and not imports['denied_imports'] and not imports['forbidden_modules_loaded']
                    and not imports['cuda_initialized'] and not imports['optimizer_steps_permitted'],
                    'Finalbank phase failed; preserve output, no automatic retry: '+phase)
                results[phase]=dict(returncode=0,imports_sha256=sha(ip),cleanup_clean=True)
            status='complete'
        elif 'readout' in profile:
            passed,results['readout']=worker('readout','software-tests')
            gate.require(passed,'New readout software tests failed; preserve receipts, no automatic retry')
            status='complete'
        else:
            # AD is first and gets an independent durable receipt before teacher work.
            if 'ad_predecessor' in profile:
                prior={}
                for key,ref in profile['ad_predecessor'].items():
                    gate.require(sha(ref['path'])==ref['sha256'],'Closed predecessor bytes changed: '+key)
                    prior[key]=json.loads(Path(ref['path']).read_text()) if key!='junit' else None
                ad=prior['result'];imports=prior['imports'];account=prior['scheduler']
                gate.require(ad['passed'] and ad['code_commit']==profile['ad']['pin']
                    and ad['counts']==dict(tests=108,failures=0,errors=0,skipped=0)
                    and ad['JUnit_sha256']==profile['ad_predecessor']['junit']['sha256']
                    and ad['imports_sha256']==profile['ad_predecessor']['imports']['sha256']
                    and not imports['denied_imports'] and not imports['forbidden_modules_loaded']
                    and imports['error'] is None and not imports['cuda_initialized']
                    and prior['terminal']['results']['ad']==ad and prior['terminal']['cleanup_error'] is None,
                    'Previous independent AD admission differs')
                gate.require(account['allocation_id']=='job-2wg5ue74mpid' and account['scheduler_state']=='Failed'
                    and account['scheduler_running_seconds']==69 and account['gpus']==0
                    and account['cpus']==8 and account['memory_gib']==64
                    and profile['scheduler_cap_seconds']==480 and 69+480+51==600,
                    'Residual CPU authority must charge69once and seal51; no new600')
                results['ad']=dict(ad,adopted_from=profile['ad_predecessor'],new_tests_run=0)
                gate.write(out/'ad-software-tests-result.json',results['ad'])
            else:
                passed,results['ad']=worker('ad','software-tests')
                gate.require(passed,'AD software qualification failed; no automatic retry')
            stage_family('teacher')
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
        terminal=dict(run_id=run,status=status,failure=failure,cleanup_error=cleanup_error,
            parent_seconds=time.monotonic()-started,phases=phases,results=results,
            scheduler_cost='pending actual terminal UI receipt',physical_actions=0,VLA_queries=0,SGD=0)
        gate.write(out/'terminal.json',terminal)
        if cleanup_error:raise RuntimeError(cleanup_error)
    if status!='complete':raise RuntimeError('Teacher software incomplete; independent AD receipt retained')

if __name__=='__main__':stage_main()
