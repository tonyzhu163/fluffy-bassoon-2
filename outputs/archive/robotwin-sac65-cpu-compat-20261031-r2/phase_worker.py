"""Owned CPU phase worker; import firewall precedes third-party site paths."""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

BANNED = ('jax', 'jaxlib', 'openpi', 'sapien', 'rlinf', 'mujoco', 'mani_skill',
          'fluffy_bassoon.rl.robotwin_noise', 'fluffy_bassoon.rl.robotwin_assets',
          'fluffy_bassoon.rl.robotwin_production', 'fluffy_bassoon.rl.metaworld_pi05',
          'fluffy_bassoon.rl.maniskill_pi05_noise', 'fluffy_bassoon.rl.robotwin_campaign_budget',
          'fluffy_bassoon.meta_rl.robotwin_trial_resume', 'fluffy_bassoon.vla', 'lerobot.policies')


def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.')
               or (prefix.endswith(('robotwin_production', 'metaworld_pi05')) and name.startswith(prefix))
               for prefix in BANNED)


class CPUImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.denied = []

    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            self.denied.append(fullname)
            raise ImportError('CPU compatibility gate forbids simulator/frozen-VLA import: ' + fullname)
        return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_source65_index(index, ctx, projection_path):
    expected = dict(complete_K4_trials=390, episodes=1560,
                    committed_source_actions=651638, selected_actions=651638)
    snapshot = ctx['tip']['checkpoint']
    iterations = index.get('source_contract', {}).get('source_iterations')
    if (index.get('partition') != 'train' or index.get('source_method') != 'sac'
            or index.get('teacher_method') != 'sac' or set(index.get('datasets', {})) != {'ad', 'dpt'}
            or any(index.get('counts', {}).get(k) != v for k, v in expected.items())
            or index.get('source_recovery_sha256') != snapshot['sha256']
            or index.get('teacher_recovery_sha256') != snapshot['sha256']
            or not isinstance(iterations, dict) or iterations.get('count') != 100
            or iterations.get('trials_per_iteration') != 6 or iterations.get('episodes_per_trial') != 4):
        raise ValueError('Wrong closed source65 index/schema/source/teacher/counters')
    inputs = index['input_sha256']
    if (inputs.get(str(Path(snapshot['path']).resolve())) != snapshot['sha256']
            or inputs.get(str(Path(projection_path).resolve())) != sha(projection_path)):
        raise ValueError('Index is not linked to this closed source65 projection/checkpoint')


def main():
    started = time.monotonic()
    guard = CPUImportGuard(); sys.meta_path.insert(0, guard)
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError('Forbidden module was loaded before the import firewall')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('tests', 'projection', 'export', 'audit'), required=True)
    p.add_argument('--context', type=Path, required=True); p.add_argument('--context-sha256', required=True)
    a = p.parse_args()
    if sha(a.context) != a.context_sha256:
        raise ValueError('Gate context changed')
    ctx = json.loads(a.context.read_text()); code, out = Path(ctx['code_root']), Path(ctx['output'])
    if sha(ctx['allocation_receipt']) != ctx['allocation_receipt_sha256']:
        raise ValueError('Actual0GPU pre-mask receipt changed')
    allocation = json.loads(Path(ctx['allocation_receipt']).read_text())
    if allocation['resource']['allocated_gpus'] != 0 or not allocation['actual_zero_gpu_attested']:
        raise ValueError('Require the actual pre-mask zero-GPU receipt')
    # -S avoids sitecustomize/.pth execution. Add only existing library paths,
    # then the exact candidate source/tests. No RLinf/OpenPI/native source path.
    site = Path(ctx['site_packages'])
    if not site.is_dir():
        raise ValueError('Existing environment site-packages path is missing')
    sys.path[:0] = [str(code / 'src'), str(code / 'tests'), str(site)]
    root = code / 'outputs/robotwin-finalbank-offline-plan-20261031-r1'
    error = None
    try:
        if a.phase in ('tests', 'export'):
            import torch
            if torch.cuda.is_initialized():
                raise RuntimeError('CUDA was initialized in a CPU-only gate')
            def no_cuda(*args, **kwargs):
                raise RuntimeError('CUDA initialization forbidden in CPU gate')
            def no_step(*args, **kwargs):
                raise RuntimeError('Optimizer steps/fits forbidden in this compatibility gate')
            torch.cuda._lazy_init = no_cuda
            torch.optim.Adam.step = torch.optim.AdamW.step = no_step
            torch.set_num_threads(4)
        if a.phase == 'tests':
            os.environ['FB2_CPU_COMPAT_READER'] = ctx['chain_reader']
            os.environ['FB2_CPU_OWNED_HELPERS'] = ctx['owned_helpers']
            import pytest
            result = pytest.main(['-q', '-p', 'no:cacheprovider',
                '--junitxml=' + str(out / 'fixtures.junit.xml'),
                str(code / 'tests/test_robotwin_history_corpus.py'),
                str(code / 'tests/test_robotwin_offline_compat_gate.py'),
                str(code / 'tests/test_robotwin_cpu_residual_bootstrap.py')])
            if result != 0:
                raise RuntimeError('Focused fixture suite failed: ' + str(result))
            suites = list(ET.parse(out / 'fixtures.junit.xml').getroot().iter('testsuite'))
            counts = {key:sum(int(s.attrib.get(key, '0')) for s in suites)
                      for key in ('tests','failures','errors','skipped')}
            if counts != dict(tests=ctx['expected_fixture_tests'], failures=0, errors=0, skipped=0):
                raise RuntimeError('Unexpected focused fixture coverage: ' + str(counts))
            (out / 'fixtures-summary.json').write_text(json.dumps(counts, indent=2) + '\n')
        elif a.phase == 'projection':
            project = load('prepared_byte_projection', root / 'project_corpus.py')
            sys.argv = [str(root / 'project_corpus.py'), '--purpose', 'reference',
                '--admission', ctx['reference_admission'], '--admission-sha256', ctx['reference_admission_sha256'],
                '--chain-reader', ctx['chain_reader'], '--output', str(out / 'private/projection'),
                '--expected-hostname', allocation['resource']['hostname']]
            project.main()
        elif a.phase == 'export':
            from fluffy_bassoon.meta_rl.robotwin_history_corpus import main as export_main
            snapshot = ctx['tip']['checkpoint']
            export_main(['--corpus-root', str(out / 'private/projection'),
                '--source-recovery', snapshot['path'], '--source-sha256', snapshot['sha256'],
                '--source-method', 'sac', '--teacher-recovery', snapshot['path'],
                '--teacher-sha256', snapshot['sha256'], '--teacher-method', 'sac',
                '--objectives', 'ad', 'dpt', '--partition', 'train',
                '--output', str(out / 'private/paired-export'), '--device', 'cpu', '--threads', '4',
                '--source-commit', ctx['code_commit'], '--expected-hostname', allocation['resource']['hostname']])
        else:
            import numpy as np
            index_path = out / 'private/paired-export/index.json'
            index = json.loads(index_path.read_text())
            verify_source65_index(index, ctx, out / 'private/projection/projection.json')
            for dataset in index['datasets'].values():
                if sha(dataset['path']) != dataset['sha256']:
                    raise ValueError('Export NPZ changed')
            with np.load(index['datasets']['ad']['path'], allow_pickle=False) as ad, \
                    np.load(index['datasets']['dpt']['path'], allow_pickle=False) as dpt:
                if set(ad.files) != set(dpt.files):
                    raise ValueError('Paired array schemas differ')
                for name in ad.files:
                    if name != 'target_noise' and not np.array_equal(ad[name], dpt[name]):
                        raise ValueError('Historical paired field changed: ' + name)
                if (not np.array_equal(ad['target_noise'], ad['executed_noise'])
                        or not np.isfinite(dpt['target_noise']).all()
                        or np.abs(dpt['target_noise']).max() > 1.500001
                        or int(ad['durations'].sum()) != 651638
                        or len(ad['durations']) != index['counts']['decisions']):
                    raise ValueError('Actual noise/label/mask-derived duration audit differs')
            receipt = dict(status='paired_CPU_compatibility_only', index_sha256=sha(index_path),
                source_rounds=65, counts=index['counts'], all_historical_paired_fields_equal=True,
                native_masks='Validated per decision by existing read_source; duration sum rechecked here',
                datasets=index['datasets'], selected_trial_prefix='included through390 globalK4 counter reconciliation',
                new_physical_actions=0, new_vla_queries=0, fits=0, SGD_updates=0,
                teacher_competence='not measured', source100='not admitted')
            (out / 'paired-audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        loaded = [name for name in sys.modules if forbidden(name)]
        torch = sys.modules.get('torch')
        cuda = torch is not None and torch.cuda.is_initialized()
        receipt = dict(phase=a.phase, error=error, seconds=time.monotonic() - started,
            denied_imports=guard.denied, forbidden_modules_loaded=loaded, cuda_initialized=cuda,
            allowed_head_scope='Torch feature-steering heads only; no frozen VLA/native runtime',
            optimizer_steps_permitted=False)
        (out / (a.phase + '-imports.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        if loaded or cuda:
            raise RuntimeError('Forbidden module/CUDA state after phase')


if __name__ == '__main__':
    main()
