"""Read terminal CPU metadata/hashes only; no tests, models or simulator."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-source-pipeline-cpu-20261031-r2')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


index, clock = read(ROOT/'index.json'), read(ROOT/'cpu-worker-clock.json')
attest = read(ROOT/'zero-gpu-attestation.json')
admission, assembly = read(ROOT/'snapshot/admission.json'), read(ROOT/'assembly/assembly.json')
manifest = read(ROOT/'assembly/topup-inputs.json')
checks = {}


def check(name, value):
    checks[name] = bool(value)
    assert value, name


check('terminal_exit0', (ROOT/'exit-code.txt').read_text().strip() == '0' and index['completed'] and clock['exit_code'] == 0)
check('actual_CPU_job_identity', index['hostname'] == attest['hostname'] == 'job-k404j8e0104x-master-0')
check('pre_mask_zero_GPU', attest['passed'] and attest['device_count_before_mask'] == 0 and not attest['cuda_initialized_before_mask'])
check('actual_CPU_memory_limits', attest['resources']['effective_cpu_quota_cores'] == 4 and attest['resources']['memory_limit_bytes'] == 16*1024**3)
check('actual_runtime_ABI', attest['python_abi'] == index['python_abi'] == 'cpython-312-x86_64-linux-gnu')
check('no_GPU_RL_or_model_work', clock['charged_gpu_seconds'] == 0 and index['simulator_actions'] == 0
    and index['optimizer_updates'] == 0 and index['model_construction'] is False and index['inference'] is False
    and index['cuda_uninitialized_at_completion'])
check('bounded_CPU_clock', clock['status'] == 'finished' and not clock['cpu_time_overrun'] and clock['charged_cpu_worker_seconds'] == 710 <= 1800)
check('attestation_and_JUnit_checksums', sha(ROOT/'zero-gpu-attestation.json') == index['zero_gpu_attestation_sha256']
    and sha(ROOT/'junit.xml') == index['junit_sha256'])
junit = ET.parse(ROOT/'junit.xml').getroot()
suites = [junit] if junit.tag == 'testsuite' else list(junit.findall('testsuite'))
test_summary = {k:sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
check('all34_metadata_tests_passed', test_summary == dict(tests=34, failures=0, errors=0, skipped=0))
check('test_capture_assemble_exit0', len(index['phases']) == 3 and all(p['completed'] and p['exit_code'] == 0 for p in index['phases']))
for name, row in index['outputs'].items():
    check(name+'_actual_file_checksum', sha(row['path']) == row['sha256'])
check('all40_fixed8_selected_once', len(admission['source_indices']['train']) == len(assembly['selected_sources']['train']) == 40
    and len(admission['source_indices']['validation']) == len(assembly['selected_sources']['validation']) == 8
    and len({r['slot'] for rows in admission['source_indices'].values() for r in rows}) == 48)
check('source_only_fullbank_admission', assembly['source_only'] and assembly['small_pilot'] is False
    and admission['source_data_from_targets'] is False and admission['additional_simulator_actions'] == 0)
check('full_cumulative_cost_preserved', manifest['cumulative_source_action_charge'] == admission['cumulative_source_action_charge'] == 457851 <= 1000000)
datasets = {}
for split, methods in assembly['datasets'].items():
    datasets[split] = {}
    for method, row in methods.items():
        p = ROOT/'assembly'/row['path']
        sidecar = read(p.with_suffix('.json'))
        check(split+'_'+method+'_actual_data_checksum', sha(p) == row['sha256'] == sidecar['sha256'])
        check(split+'_'+method+'_actual_manifest_rows', sidecar['trials'] == row['trials'] and sidecar['transitions'] == row['transitions'])
        datasets[split][method] = dict(row, path=str(p), sidecar_sha256=sha(p.with_suffix('.json')))
bridge = [r for r in admission['source_indices']['train'] if r['slot'] in ('libero_spatial:00/train','libero_object:00/train')]
check('latest_bridge_prefixes_selected', sorted((r['selected_cases'],r['selected_chunks'],r['selected_teacher_updates']) for r in bridge)
    == [(33,1987,29940),(55,1927,28620)])
print(json.dumps(dict(format='source40_CPU_capture_assembly_saved_audit_v1', observed_at='2026-10-31 Asia/Shanghai',
    job_id='job-k404j8e0104x', run_id=ROOT.name, source_commit=index['source_commit'], checks=checks,
    index_sha256=sha(ROOT/'index.json'), clock_sha256=sha(ROOT/'cpu-worker-clock.json'),
    test_summary=test_summary, allocated_attestation=dict(path=str(ROOT/'zero-gpu-attestation.json'),
        sha256=index['zero_gpu_attestation_sha256'], python_abi=attest['python_abi'], torch=attest['torch_version'],
        pre_mask_GPU_count=0, CUDA_initialized=False, cpu_quota=4, memory_bytes=16*1024**3),
    CPU_seconds=710, GPU_seconds=0, simulator_actions=0, outputs=index['outputs'], datasets=datasets,
    cumulative_source_action_charge=admission['cumulative_source_action_charge'], selected_bridge_prefixes=bridge,
    phase_seconds={p['phase']:p['ended_unix']-p['started_unix'] for p in index['phases']},
    predecessor_failure=dict(run='libero40-source-pipeline-cpu-20261031-r1', job='job-ujyjxq64rql8',
        GPU_count=0, scheduler_seconds=49, source_prefix='0478', cause='Runtime guard assumedPython3.10; actual qualified binary is3.12.15',
        evidence='Root verified UI/runtime metadata; no output absence is used as job-status evidence'),
    limitations='Immutable source data is ready for fitting; no controller training, model checkpoint or robot performance result is established by this CPU qualification.'), indent=2, allow_nan=False))
