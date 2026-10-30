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

RUN = 'robotwin-sac65-cpu-compat-20261031-r2'
ANCESTRY_SHA = '5e65bc308c8d73ab21ff80452c39fec98b873171e49a1432312a8645bd07d861'
READER_PIN = 'c62be99615e797fbec108e1f1b5e232e4a591494'
READER_PATH = 'outputs/robotwin-trained-handoff-control-20261030-r1/control.py'
READER_SHA = '67db52fa8f6fa4e8bb56bc6e5cea9f531b5fe41716845350f3372226c5de77a0'
TIP_SHA = '8c429b7d7a66b1374ad95a1136b783717509eef0651f457d775c39b59d5f75b6'
BUDGET_SHA = 'b0dddb34275602ddc7bb9fb6aba764285925115ba87d44c028d70ba38f8c56f8'
PREDECESSOR_SHA = '8e26ce12a16cc63a8636c345e74c9e453a153945050a447d6ae29e74eebf3260'
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
    directory = Path(code) / 'outputs' / RUN
    raw = (directory / 'budget.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BUDGET_SHA, 'Residual budget bytes changed')
    budget = json.loads(raw)
    expected = dict(original_scheduler_seconds=1800, charged_scheduler_seconds=149,
        residual_scheduler_seconds=1651, scheduler_cap_seconds=1500, global_work_seconds=1400,
        external_TERM_seconds=1470, external_KILL_seconds=1480, cleanup_reserve_seconds=30,
        bootstrap_wait_seconds=120, sealed_unused_seconds=151, gpus=0, cpus=8, memory_gib=64)
    require(budget.get('schema') == 'robotwin_cpu_residual_budget_v1' and budget.get('run_id') == RUN
        and all(type(budget.get(k)) is int and budget[k] == v for k,v in expected.items()),
        'Wrong original charge, residual cap or timer agreement')
    prior = (directory / 'predecessor-scheduler-accounting.json').read_bytes()
    require(hashlib.sha256(prior).hexdigest() == PREDECESSOR_SHA
        and budget.get('predecessor_accounting_sha256') == PREDECESSOR_SHA,
        'Prior scheduler cost receipt changed')
    evidence = json.loads(prior)
    require(evidence['allocation_id'] == budget['predecessor_allocation'] == 'job-eahrz09v6kkk'
        and budget['predecessor_code_commit'] == '29df647fd30469fc04743baf47ba331718c9d1a2'
        and evidence['scheduler_state'] == 'Failed' and evidence['scheduler_running_seconds'] == 149
        and evidence['resource_envelope_seconds'] == 1800 and evidence['residual_seconds'] == 1651
        and evidence['phases'] == [], 'Not the exact diagnosed predecessor charge')
    require(149 + 1500 + 151 == 1800, 'Residual envelope arithmetic differs')
    return dict(path=str(directory/'budget.json'), sha256=BUDGET_SHA,
                predecessor_sha256=PREDECESSOR_SHA, values=budget)


def main():
    started = time.monotonic()  # Before resource/Git/ancestry/import/tests/projection/export.
    deadline = started + 1400
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
    require(not out.exists() and out == Path('/mowenjie/fluffy-bassoon-2/outputs', RUN).resolve()
            and not out.is_relative_to(code), 'Use this fresh unique runtime output outside source code')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'private').mkdir(); (out / 'tmp').mkdir()
    sys.stdout = sys.stderr = (out / 'parent.log').open('a', buffering=1)
    write(out / 'intent.json', dict(run=RUN, code_commit=a.code_commit, hostname=hostname,
          global_work_seconds=1400, external_TERM_seconds=1470, external_KILL_seconds=1480,
          scheduler_cap_seconds=1500, budget=budget, physical_actions=0, labels='compatibility only', fits=0))
    def interrupted(signum, frame):
        raise RuntimeError('Owned CPU gate received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    status, failure, runtime = 'incomplete', None, None
    phases = []
    try:
        if a.bootstrap_ticket is not None:
            write(out / 'bootstrap-intent.json', dict(ticket_path=str(a.bootstrap_ticket), max_wait_seconds=120,
                  code_commit=a.code_commit, hostname=hostname, output=str(out),
                  work_elapsed_seconds=time.monotonic() - started, work_deadline_seconds_from_start=1400,
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
        if shutil.which('nvidia-smi'):
            log = out / 'visibility.log'
            smi = runtime.run('visibility', ['nvidia-smi', '-L'], code, os.environ, log, 5)
            observed = dict(returncode=smi['returncode'], stdout=log.read_text(), stderr='combined raw log')
        else:
            observed = dict(returncode=None, stdout='', stderr='nvidia-smi absent')
        devices = [str(x) for x in Path('/dev').glob('nvidia[0-9]*')]
        require(not devices and not any(line.startswith('GPU ') for line in observed['stdout'].splitlines()),
                'A GPU is visible before masking; refuse to disguise allocation/visibility')
        visible = before.get('NVIDIA_VISIBLE_DEVICES')
        require(visible in (None, '', 'none', 'void'), 'Nonempty GPU namespace before masking')
        require('torch' not in sys.modules and 'jax' not in sys.modules, 'Parent must remain stdlib-only')
        write(out / 'allocation-before-mask.json', dict(resource=resource, resource_sha256=a.resource_sha256,
              environment_before_mask=before, device_nodes=devices, nvidia_smi=observed,
              torch_imported=False, actual_zero_gpu_attested=True))
        # Mask only AFTER the actual allocation receipt and independent visibility observation.
        os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1',
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          OPENBLAS_NUM_THREADS='4', TMPDIR=str(out / 'tmp'))
        for name, command in [('git-head', ['git', 'rev-parse', 'HEAD']),
                              ('git-status', ['git', '--no-optional-locks', '-c', 'core.fsmonitor=false',
                                              'status', '--porcelain'])]:
            result = runtime.run(name, command, code, os.environ, out / (name + '.log'), 10)
            require(result['returncode'] == 0, 'Candidate Git identity check failed')
        head = (out / 'git-head.log').read_text().strip()
        dirty = (out / 'git-status.log').read_text().strip()
        require(head == a.code_commit and not dirty, 'Exact clean candidate pin required')
        reader = out / 'private' / 'qualified-c62-reader.py'
        result = runtime.run('c62-reader', ['git', '--no-pager', 'show', READER_PIN + ':' + READER_PATH],
                             code, os.environ, reader, 10)
        require(result['returncode'] == 0, 'Qualified reader Git extraction failed')
        require(sha(reader) == READER_SHA, 'Qualified reader Git object differs')
        ancestry_file = code / 'outputs/robotwin-finalbank-offline-plan-20261031-r1/SAC65-REFERENCE.json'
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
        context = dict(code_root=str(code), code_commit=a.code_commit, output=str(out),
              expected_fixture_tests=121,
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
            remaining = deadline - time.monotonic() - 30  # Owned cleanup is inside the same1400s.
            reserve = {'tests': 900, 'projection': 660, 'export': 90, 'audit': 0}[phase]
            maximum = min({'tests': 180, 'projection': 900, 'export': 900, 'audit': 90}[phase], remaining - reserve)
            require(maximum >= {'tests': 30, 'projection': 60, 'export': 540, 'audit': 15}[phase],
                    'Insufficient residual time; retain incomplete evidence, no extra1500s')
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
