"""One owner-launched CPU gate. Stdlib only; no submission or source controls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

RUN = 'robotwin-sac65-cpu-compat-20261031-r5'
ANCESTRY_SHA = '5e65bc308c8d73ab21ff80452c39fec98b873171e49a1432312a8645bd07d861'
READER_PIN = 'c62be99615e797fbec108e1f1b5e232e4a591494'
READER_PATH = 'outputs/robotwin-trained-handoff-control-20261030-r1/control.py'
READER_SHA = '67db52fa8f6fa4e8bb56bc6e5cea9f531b5fe41716845350f3372226c5de77a0'
TIP_SHA = '8c429b7d7a66b1374ad95a1136b783717509eef0651f457d775c39b59d5f75b6'
BUDGET_SHA = '6771bcb6459142b90245241613cf81501d2995ae438eab92db59f606a8d0dec4'
CHECKPOINT_SHA = 'b09c13f54e4db8994c9a5d94cd6f18b6af3514d4d5ef4e63f1f91e482578b0e0'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def wait_bootstrap(ticket_path, *, code_commit, output, hostname, deadline, clock=None, sleep=None):
    """Wait only for an owner-uploaded actual-allocation ticket; never reset budget."""
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    began = clock()
    stop = min(deadline, began + 120)
    while not ticket_path.exists():
        remaining = stop - clock()
        if remaining <= 0:
            raise TimeoutError('Actual resource bootstrap ticket did not arrive within shared deadline')
        sleep(min(1., remaining))
    require(clock() < stop, 'Bootstrap ticket arrived after the shared wait deadline')
    require(ticket_path.is_file() and not ticket_path.is_symlink(), 'Bootstrap ticket must be a regular owner file')
    raw = ticket_path.read_bytes()
    ticket = json.loads(raw)
    require(ticket.get('schema') == 'robotwin_cpu_gate_actual_bootstrap_v1'
            and ticket.get('run_id') == RUN and ticket.get('code_commit') == code_commit
            and ticket.get('hostname') == hostname
            and ticket.get('output') == str(output), 'Stale or mismatched actual resource bootstrap ticket')
    ref = ticket['resource_receipt']
    require(Path(ref['path']).is_absolute() and len(ref['sha256']) == 64,
            'Bootstrap must name the actual absolute resource receipt and SHA')
    return ref, dict(ticket_path=str(ticket_path), ticket_sha256=hashlib.sha256(raw).hexdigest(),
                     ticket=ticket, wait_seconds=clock() - began, remaining_work_seconds=deadline - clock())





def load_budget(code):
    directory=Path(code)/'outputs'/RUN;raw=(directory/'budget.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==BUDGET_SHA,'Separate CPU grant bytes changed')
    budget=json.loads(raw)
    require(budget.get('schema')=='robotwin_separate_cpu_qualification_grant_v1' and budget.get('run_id')==RUN,
            'Wrong separately authorized qualification grant')
    closed=budget['closed_old_envelope']
    require(closed['status']=='CLOSED' and closed['scheduler_limit_seconds']==1800
        and closed['actual_scheduler_seconds']==620 and closed['unused_sealed_seconds']==1180
        and 620+1180==1800,'The old CPU envelope must stay closed and fully accounted')
    wanted=[('job-eahrz09v6kkk',149,'8e26ce12a16cc63a8636c345e74c9e453a153945050a447d6ae29e74eebf3260'),
            ('job-wcfjtype0eec',150,'2f60430cc856aa8615b6a1e98c197b88058fae96f64bc91a61f061da223f6087'),
            ('job-0mb2yh59u58h',150,'5aadbf67abb219bc44b5411454b2f2b103fa63f87ab583ab1830a0e7821ead50'),
            ('job-tw0hcz9ds2gy',171,'77e90aeeee44352a36de9a6bcffab275e3005aa161bbb33ff113a3a1057b22c9')]
    require(len(closed['predecessors'])==4,'Need all four actual charges')
    total=0
    for number,(ref,(allocation,cost,expected_sha)) in enumerate(zip(closed['predecessors'],wanted,strict=True),1):
        require(ref['file']=='predecessor-r'+str(number)+'-accounting.json','Wrong old receipt path')
        prior=(directory/ref['file']).read_bytes();evidence=json.loads(prior)
        require(hashlib.sha256(prior).hexdigest()==ref['sha256']==expected_sha
            and evidence['allocation_id']==ref['allocation_id']==allocation
            and evidence['scheduler_running_seconds']==ref['scheduler_seconds']==cost
            and evidence['scheduler_state']=='Failed' and evidence['exit_code']==1
            and evidence['gpus']==0 and evidence['cpus']==8 and evidence['memory_gib']==64
            and evidence['phases']==[],'Closed old allocation accounting differs')
        total+=cost
    require(total==620,'Old charges must never be refunded or reset')
    grant=budget['new_qualification_grant']
    expected=dict(scheduler_cap_seconds=1800,global_work_seconds=1700,external_TERM_seconds=1770,
        external_KILL_seconds=1780,cleanup_reserve_seconds=30,bootstrap_wait_seconds=120,
        local_code_stage_seconds=30,gpus=0,cpus=8,memory_gib=64,old_budget_credit_seconds=0,
        borrowed_old_sealed_seconds=0)
    require(grant['id']==RUN and all(type(grant.get(k)) is int and grant[k]==v for k,v in expected.items())
        and grant['automatic_resubmission'] is False and grant['actual_allocation_id'] is None
        and grant['actual_scheduler_seconds'] is None,'Wrong separate grant, resource or clock contract')
    return dict(path=str(directory/'budget.json'),sha256=BUDGET_SHA,values=budget)


def pre_mask_visibility(runtime,code,out,environment):
    worker=code/'outputs'/RUN/'visibility_probe.py';receipt=out/'visibility-probe.json'
    result=runtime.run('visibility',[sys.executable,'-I','-B','-S',str(worker),'--output',str(receipt)],
                       code,environment,out/'visibility.log',5)
    require(result['returncode']==0 and result['cleanup']['clean'],'Visibility worker or owned cleanup failed')
    observation=json.loads(receipt.read_text())
    require(observation.get('schema')=='robotwin_pre_mask_visibility_v1'
        and observation.get('allocated_GPU_count_inferred') is False,'Invalid visibility diagnostic receipt')
    spec=importlib.util.spec_from_file_location('saved_cpu_visibility_validator',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate(observation)
    return observation,dict(path=str(receipt),sha256=sha(receipt))


def main():
    started = time.monotonic()  # Before resource/Git/ancestry/import/tests/projection/export.
    deadline = started + 1700
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'output', 'owned-helpers'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--code-commit', required=True)
    p.add_argument('--bootstrap-ticket', type=Path)
    p.add_argument('--resource-receipt', type=Path)
    p.add_argument('--resource-sha256')
    p.add_argument('--expected-hostname')
    a = p.parse_args()
    budget = load_budget(a.code_root)
    hostname = socket.gethostname()
    require(sys.platform == 'linux', 'Only an allocated Linux host')
    if a.bootstrap_ticket is not None:
        require(all(value is None for value in (a.resource_receipt, a.resource_sha256, a.expected_hostname)),
                'Bootstrap and already-bound direct receipt modes cannot be mixed')
    else:
        require(a.resource_receipt is not None and a.resource_sha256 is not None
                and a.expected_hostname == hostname, 'Direct mode needs exact actual host/receipt/SHA')
    out, code = a.output.resolve(), a.code_root.resolve()
    deployed_code=code
    require(not out.exists() and out == Path('/mowenjie/fluffy-bassoon-2/outputs', RUN).resolve()
            and not out.is_relative_to(code), 'Use this fresh unique runtime output outside source code')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'private').mkdir(); (out / 'tmp').mkdir()
    sys.stdout = sys.stderr = (out / 'parent.log').open('a', buffering=1)
    write(out / 'intent.json', dict(run=RUN, code_commit=a.code_commit, hostname=hostname,
          global_work_seconds=1700, external_TERM_seconds=1770, external_KILL_seconds=1780,
          scheduler_cap_seconds=1800, budget=budget, deployed_code_root=str(deployed_code), physical_actions=0, labels='compatibility only', fits=0))
    def interrupted(signum, frame):
        raise RuntimeError('Owned CPU gate received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    status, failure, runtime = 'incomplete', None, None
    phases = []
    try:
        if a.bootstrap_ticket is not None:
            write(out / 'bootstrap-intent.json', dict(ticket_path=str(a.bootstrap_ticket), max_wait_seconds=120,
                  code_commit=a.code_commit, hostname=hostname, output=str(out),
                  work_elapsed_seconds=time.monotonic() - started, work_deadline_seconds_from_start=1700,
                  bootstrap_expires_at_unix=time.time()+120, run_id=RUN, budget=budget))
            ref, bootstrap = wait_bootstrap(a.bootstrap_ticket, code_commit=a.code_commit,
                output=out, hostname=hostname, deadline=deadline)
            write(out / 'bootstrap-receipt.json', bootstrap)
            a.resource_receipt, a.resource_sha256 = Path(ref['path']), ref['sha256']
        resource_bytes = a.resource_receipt.read_bytes()
        require(hashlib.sha256(resource_bytes).hexdigest() == a.resource_sha256, 'Actual resource receipt changed')
        resource = json.loads(resource_bytes)
        require(resource['schema'] == 'robotwin_actual_cpu_allocation_v1'
                and resource['hostname'] == hostname
                and hostname.startswith(resource['allocation_id'] + '-')
                and type(resource['allocated_gpus']) is int and resource['allocated_gpus'] == 0
                and resource['allocated_cpus'] == 8 and resource['memory_gib'] == 64
                and resource['observed_running_allocation'] is True,
                'Need the actual0GPU/8CPU/64GiB allocation, not a requested-resource template')
        require(sha(resource['proof']['path']) == resource['proof']['sha256'], 'Allocation UI proof changed')
        adapter_path = code / 'outputs' / RUN / 'owned_cpu_runtime.py'
        spec = importlib.util.spec_from_file_location('owned_cpu_phase_runtime', adapter_path)
        adapter = importlib.util.module_from_spec(spec); sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        runtime = adapter.CPUPhaseRuntime(a.owned_helpers, out / 'private/ownership', deadline=deadline)
        before = {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                  'SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_GPUS_ON_NODE')}
        observed,visibility_receipt=pre_mask_visibility(runtime,code,out,os.environ)
        require(observed['environment_before_mask']==before,'Visibility environment changed before masking')
        require('torch' not in sys.modules and 'jax' not in sys.modules, 'Parent must remain stdlib-only')
        write(out / 'allocation-before-mask.json', dict(resource=resource, resource_sha256=a.resource_sha256,
              environment_before_mask=before, device_nodes=observed['device_nodes'],
              nvidia_smi=observed['nvidia_smi'], visibility_worker=visibility_receipt,
              torch_imported=False, actual_zero_gpu_attested=True))
        # Mask only AFTER the actual allocation receipt and independent visibility observation.
        os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1',
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          OPENBLAS_NUM_THREADS='4', TMPDIR=str(out / 'tmp'))
        stage_cap=min(30,deadline-time.monotonic()-960)
        require(stage_cap>=10,'Insufficient setup time to stage exact code while preserving test floor')
        staging=deployed_code/'outputs'/RUN/'stage_local_code.py'
        stage_spec=importlib.util.spec_from_file_location('exact_local_git_environment',staging)
        stage_module=importlib.util.module_from_spec(stage_spec);stage_spec.loader.exec_module(stage_module)
        clean_environment=stage_module.git_environment(os.environ)
        # Apply the same filter to qualified Git extraction and all later
        # consumers, including Git invoked inside phase-worker libraries.
        for key in tuple(os.environ):
            if key not in clean_environment:os.environ.pop(key)
        result=runtime.run('local-code',[sys.executable,'-I','-B','-S',str(staging),
            '--source-code',str(deployed_code),'--code-commit',a.code_commit,'--output',str(out)],
            deployed_code,os.environ,out/'local-code-stage.log',stage_cap)
        require(result['returncode']==0 and result['cleanup']['clean'],'Exact local-code staging or cleanup failed')
        local_code=json.loads((out/'local-code.json').read_text())
        require(local_code['completed'] and local_code['code_commit']==a.code_commit
            and local_code['original_head']==local_code['effective_head']==a.code_commit
            and local_code['original_code_root']==str(deployed_code)
            and local_code['full_clean_git_status'],'Local exact-code receipt differs')
        code=Path(local_code['effective_code_root'])
        require(code.is_dir() and code.resolve()==code and not code.is_symlink(),'Missing/redirected job-local code')
        (out/'git-head.log').write_text(a.code_commit+'\n')
        (out/'git-status.log').write_bytes(Path(local_code['status_path']).read_bytes())
        require(sha(out/'git-status.log')==local_code['status_sha256'],'Full local Git status changed')
        reader = out / 'private' / 'qualified-c62-reader.py'
        result = runtime.run('c62-reader', ['git', '--no-pager', 'show', READER_PIN + ':' + READER_PATH],
                             code, os.environ, reader, 10)
        require(result['returncode'] == 0, 'Qualified reader Git extraction failed')
        require(sha(reader) == READER_SHA, 'Qualified reader Git object differs')
        ancestry_file = deployed_code / 'outputs/robotwin-finalbank-offline-plan-20261031-r1/SAC65-REFERENCE.json'
        require(sha(ancestry_file) == ANCESTRY_SHA, 'Wrong closed-prefix ancestry')
        ancestry = json.loads(ancestry_file.read_text())
        require(ancestry['root_count'] == 26 and ancestry['tip_manifest']['sha256'] == TIP_SHA,
                'Not the pinned closed26-root SAC65 chain')
        tip = json.loads(Path(ancestry['tip_manifest']['path']).read_text())
        require(sha(ancestry['tip_manifest']['path']) == TIP_SHA
                and tip['checkpoint']['sha256'] == CHECKPOINT_SHA
                and tip['expected']['completed_source_iterations'] == 65
                and tip['expected']['completed_source_trials'] == 390
                and tip['expected']['source_steps'] == 651638, 'Wrong closed65 boundary')
        # Stat-only inventory precedes bulk hashes/model imports. It is bounded,
        # records actual tiny-file/byte counts and proves hardlink device assumptions.
        inventory_deadline = min(deadline - 900, time.monotonic() + 180)
        free = shutil.disk_usage(out).free
        require(free >= 16 << 30, 'Need16GiB free for indexes/metadata/temporary files')
        probe = out / 'private' / 'write-probe'
        with probe.open('xb') as stream:
            stream.write(b'cpu-gate-write-probe'); stream.flush(); os.fsync(stream.fileno())
        probe.unlink()
        roots, count, total, tiny4, tiny64 = [], 0, 0, 0, 0
        device = out.stat().st_dev
        for name in ancestry['ordered_corpus_roots']:
            root = Path(name).resolve(); n = size = 0
            for folder in ('episodes', 'frames', 'policies'):
                top = root / folder
                require(top.is_dir() and not top.is_symlink(), 'Redirected/missing corpus folder')
                for path in top.rglob('*'):
                    require(time.monotonic() < inventory_deadline, 'Inventory phase time exhausted')
                    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Raw path escape')
                    if path.is_dir():
                        continue
                    st = path.stat()
                    require(path.is_file() and st.st_dev == device, 'Hardlink-only gate requires one filesystem')
                    n += 1; size += st.st_size; count += 1; total += st.st_size
                    tiny4 += st.st_size <= 4096; tiny64 += st.st_size <= 65536
                    require(count <= 500000 and total <= 256 << 30, 'Conservative I/O eligibility cap exceeded')
            roots.append(dict(root=str(root), files=n, bytes=size))
        write(out / 'io-inventory.json', dict(roots=roots, files=count, bytes=total,
              files_le4KiB=tiny4, files_le64KiB=tiny64, same_filesystem_device=device,
              free_bytes=free, hardlink_only=True, copy_fallback=False,
              conservative_full_read_estimate_bytes=8 * total,
              estimate_scope='8x byte estimate only; many tiny-file opens/CPU archive checks also cost time'))
        admission = dict(assembly_cpu_gate_passed=False, closed_source100_authorized=False,
              closed_reference_authorized=True, expected_source_rounds=65, method='sac',
              lineage_commit=ancestry['lineage_commit'],
              ancestry=dict(path=str(ancestry_file), sha256=ANCESTRY_SHA))
        admission_file = out / 'private' / 'reference-admission.json'
        write(admission_file, admission)
        context = dict(code_root=str(code), deployed_code_root=str(deployed_code),
              local_code_receipt=str(out/'local-code.json'), code_commit=a.code_commit, output=str(out),
              expected_fixture_tests=150,
              owned_helpers=str(a.owned_helpers.resolve()),
              site_packages=str(Path(sys.executable).parent.parent / 'lib' /
                                ('python%d.%d' % sys.version_info[:2]) / 'site-packages'),
              ancestry=dict(path=str(ancestry_file), sha256=ANCESTRY_SHA), tip=tip,
              chain_reader=str(reader), reference_admission=str(admission_file),
              reference_admission_sha256=sha(admission_file),
              allocation_receipt=str(out / 'allocation-before-mask.json'),
              allocation_receipt_sha256=sha(out / 'allocation-before-mask.json'))
        context_file = out / 'private' / 'context.json'; write(context_file, context)
        worker = code / 'outputs' / RUN / 'phase_worker.py'
        for phase in ('tests', 'projection', 'export', 'audit'):
            remaining = deadline - time.monotonic() - 30  # Owned cleanup is inside the same1700s.
            reserve = {'tests': 900, 'projection': 660, 'export': 90, 'audit': 0}[phase]
            maximum = min({'tests': 180, 'projection': 900, 'export': 900, 'audit': 90}[phase], remaining - reserve)
            require(maximum >= {'tests': 30, 'projection': 60, 'export': 540, 'audit': 15}[phase],
                    'Insufficient residual time; retain incomplete evidence, no extra1800s')
            receipt = dict(phase=phase, status='intent', cap_seconds=maximum,
                           elapsed_before=time.monotonic() - started)
            write(out / (phase + '-phase.json'), receipt)
            phase_started = time.monotonic()
            try:
                result = runtime.run(phase, [sys.executable, '-I', '-B', '-S', str(worker), '--phase', phase,
                    '--context', str(context_file), '--context-sha256', sha(context_file)], code,
                    os.environ, out / (phase + '.log'), maximum)
            except BaseException as exc:
                receipt.update(status='timeout_or_execution_error', error=type(exc).__name__,
                               seconds=time.monotonic() - phase_started,
                               log_sha256=sha(out / (phase + '.log')))
                write(out / (phase + '-phase.json'), receipt); phases.append(receipt)
                raise
            receipt.update(status='passed' if result['returncode'] == 0 else 'failed',
                           returncode=result['returncode'], cleanup=result['cleanup'], seconds=time.monotonic() - phase_started,
                           log_sha256=sha(out / (phase + '.log')))
            write(out / (phase + '-phase.json'), receipt); phases.append(receipt)
            require(result['returncode'] == 0, 'Phase failed: ' + phase)
        require(time.monotonic() < deadline, 'Global gate time exhausted')
        runtime.close()
        status = 'passed_closed_SAC65_CPU_compatibility_only'
        write(out / 'COMPATIBILITY-PASS.json', dict(status=status, code_commit=a.code_commit,
              source_rounds=65, complete_K4=390, source_actions=651638, phases=phases,
              additional_robot_actions=0, additional_vla_queries=0, fits=0, actual_gpus=0,
              finalbank_admitted=False, teacher_competence_measured=False))
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        write(out / 'INCOMPLETE.json', dict(status='not_published_not_admitted', failure=failure))
        raise
    finally:
        # No more work is launched. Preserve one bounded ownership cleanup even
        # when a second TERM/INT arrives during the parent's wait.
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                status = 'incomplete'
                failure = (failure or '') + ' owned-cleanup: ' + str(exc)
                write(out / 'OWNED-CLEANUP-FAILED.json', dict(error=str(exc), qualified=False))
        write(out / 'terminal.json', dict(status=status, failure=failure, phases=phases,
              elapsed_seconds=time.monotonic() - started, actual_scheduler_seconds='owner_terminal_receipt_required',
              additional_robot_actions=0, fits=0))


if __name__ == '__main__':
    main()
