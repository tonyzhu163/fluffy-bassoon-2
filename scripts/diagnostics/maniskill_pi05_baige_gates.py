"""Finite native motor/headroom admission; no learner training or automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


NATIVE_TESTS = ('tests/test_core_imports.py', 'tests/test_maniskill_pi05.py',
                'tests/test_rlinf_checkpoint_compat.py')
CORE_TESTS = (
    'tests/test_ctm_sync_options.py', 'tests/test_meta_rl_ppo.py',
    'tests/test_continuous_memory_machine.py', 'tests/test_meta_rl_cmm.py',
    'tests/test_meta_rl_fewshot.py', 'tests/test_meta_rl_copy.py',
    'tests/test_meta_rl_libero.py', 'tests/test_meta_rl_native_prefix.py',
    'tests/test_meta_rl_train.py')


def run_tests(root, output, name, files, *, environment=None):
    command = [sys.executable, '-m', 'pytest', '-q', *files]
    with (output/(name+'.log')).open('w') as stream:
        try:
            result = subprocess.run(command, cwd=root, stdout=stream,
                stderr=subprocess.STDOUT, env=environment, timeout=600)
            code = result.returncode
        except subprocess.TimeoutExpired:
            stream.write('\nQualification exceeded its fixed 600-second execution cap.\n')
            code = 124
    return dict(command=command, return_code=code, passed=code == 0,
                log=name+'.log')


def reuse_core_cpu(root, qualified_root, receipt):
    """Reuse allocated CPU admission only for byte-identical core code and tests."""
    qualified_source = '81caf220e4ad9989859436073fdb2585a24b1d60'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=qualified_root, text=True).strip() != qualified_source:
        raise ValueError('Unexpected qualified CPU source')
    previous = read(receipt)['core_cpu']
    if not previous['passed'] or previous['return_code'] != 0:
        raise ValueError('Prior allocated core CPU qualification did not pass')
    files = subprocess.check_output(['git', 'ls-files', 'src/fluffy_bassoon/meta_rl',
        'src/lerobot_policy_ctm', 'src/fluffy_bassoon/gpu_execution.py', *CORE_TESTS],
        cwd=qualified_root, text=True).splitlines()
    hashes = {}
    for name in files:
        if not name.endswith('.py'):
            continue
        before, after = (qualified_root/name).read_bytes(), (root/name).read_bytes()
        if before != after:
            raise ValueError('Core CPU proof cannot be reused after changing '+name)
        hashes[name] = hashlib.sha256(after).hexdigest()
    return dict(passed=True, return_code=0, reused=True, source_commit=qualified_source,
        receipt=str(receipt), receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        files_sha256=hashes, previous=previous)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--rlinf-root', required=True, type=Path)
    p.add_argument('--rlinf-loader-sha256', required=True)
    p.add_argument('--scope', choices=('all', 'native', 'cmm'), default='all')
    p.add_argument('--qualified-core-worktree', type=Path)
    p.add_argument('--qualified-core-receipt', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require exact clean source')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'src'))
    from fluffy_bassoon.rl.maniskill_pi05 import execution_mode, job_id, prepare_gpu
    assert execution_mode() == 'baige'
    base = Path('/mowenjie/fluffy-bassoon')
    rlinf = args.rlinf_root
    loader_sha = hashlib.sha256((rlinf/'rlinf/models/embodiment/openpi/checkpoint.py').read_bytes()).hexdigest()
    if loader_sha != args.rlinf_loader_sha256:
        raise ValueError('Require the reviewed immutable native checkpoint loader')
    rl = base/'inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor'
    sft = base/'inputs/rlinf-pi05-maniskill-sft-eb0e2c90-20261028-r1'
    rl_norm = rl/'assets/global_step_150/meta/norm_stats.json'
    sft_norm = sft/'physical-intelligence/maniskill/norm_stats.json'
    rl_sha = '0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1'
    sft_sha = '9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89'
    phases = [('gate-rl-smoke', rl, rl_norm, rl_sha, 2, 10),
              ('gate-rl', rl, rl_norm, rl_sha, 20, 50),
              ('gate-sft', sft, sft_norm, sft_sha, 20, 50),
              ('headroom-rl', rl, rl_norm, rl_sha, None, 64)]
    write(args.output_dir/'contract.json', dict(source_commit=args.source_commit, job_id=job_id(),
        rlinf_commit='e1a82ebf0c1af69da6ab9b1519f5f3692fe33566',
        rlinf_root=str(rlinf), rlinf_checkpoint_loader_sha256=loader_sha,
        phases=[dict(name=n, episodes_per_object=e, num_envs=b) for n, _, _, _, e, b in phases],
        scope=args.scope, total_episodes=0 if args.scope == 'cmm' else 1754,
        native_steps_per_episode=80, learner_training=False,
        test_scope=dict(native=NATIVE_TESTS, recurrent_core=CORE_TESTS,
            libero_integration='tests/test_libero_observation_integration.py requires the separate qualified LIBERO environment'),
        calibration_smoke=dict(variant='cmm', shots=20, seed=0, epochs=1,
                               max_elapsed_seconds=600, full_calibration_launched=False),
        rl_norm_sha256=hashlib.sha256(rl_norm.read_bytes()).hexdigest(),
        sft_norm_sha256=hashlib.sha256(sft_norm.read_bytes()).hexdigest(),
        norm_stats_equal=read(rl_norm) == read(sft_norm)))
    try:
        tests = {}
        if args.scope != 'cmm':
            tests['native_cpu'] = run_tests(root, args.output_dir, 'cpu-native-tests', NATIVE_TESTS)
        write(args.output_dir/'tests.json', tests)
        if args.scope != 'cmm' and not tests['native_cpu']['passed']:
            raise RuntimeError('Native motor CPU admission failed; inspect cpu-native-tests.log')
        # Recurrent learners are a separate method from the frozen native motor.
        # Keep their result without making an unrelated failure block motor evidence.
        if args.scope != 'native':
            if not args.qualified_core_worktree or not args.qualified_core_receipt:
                raise ValueError('Require explicit unchanged allocated core CPU proof')
            tests['core_cpu'] = reuse_core_cpu(root, args.qualified_core_worktree, args.qualified_core_receipt)
        write(args.output_dir/'tests.json', tests)
        prepare_gpu(root, args.output_dir)
        results = []
        for name, model, norm, expected, episodes, batch in ([] if args.scope == 'cmm' else phases):
            write(args.output_dir/'progress.json', dict(phase=name, completed_phases=len(results), total_phases=4))
            script = 'maniskill_pi05_rotation_cache.py' if episodes is None else 'maniskill_pi05_object_gate.py'
            command = [sys.executable, str(root/'scripts/diagnostics'/script),
                '--rlinf-root', str(rlinf), '--model-path', str(model), '--norm-stats', str(norm),
                '--expected-model-sha256', expected, '--source-commit', args.source_commit,
                '--output-dir', str(args.output_dir/name), '--num-envs', str(batch)]
            command.extend(['--scenes-per-object', '4'] if episodes is None else ['--episodes-per-object', str(episodes)])
            with (args.output_dir/(name+'.log')).open('w') as stream:
                subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=5400)
            phase = args.output_dir/name
            summary, contract, probe = [read(phase/file) for file in ('summary.json','contract.json','gpu-probe.json')]
            assert summary['completed'] and summary['source_commit'] == args.source_commit
            assert contract['job_id'] == job_id() and contract['model_sha256'] == expected
            assert contract['rlinf_checkpoint_loader_sha256'] == loader_sha
            assert probe['status'] == 'passed' and probe['hostname'] == job_id()+'-master-0'
            binding = read(phase/'renderer-binding.json')
            assert binding['job_id'] == job_id()
            assert binding['cuda_pci_bus_id'] == probe['devices'][0]['pci_bus_id']
            assert summary['frozen_model_before'] == summary['frozen_model_after'] == contract['frozen_model_sha256']
            rows = [json.loads(line) for line in (phase/('outcomes.jsonl' if episodes is None else 'episodes.jsonl')).read_text().splitlines()]
            issued = [json.loads(line) for line in (phase/'issued-actions.jsonl').read_text().splitlines()]
            assert len(rows) == (704 if episodes is None else 25*episodes)
            assert sum(x['native_actions'] for x in issued) == len(rows)*80 == summary['native_actions']
            assert all(x['gripper_rotation_z_unchanged'] and x['xy_norm_max_error'] < 1e-5 for x in issued)
            results.append(dict(name=name, episodes=len(rows), native_actions=summary['native_actions'],
                summary_sha256=hashlib.sha256((phase/'summary.json').read_bytes()).hexdigest(), summary=summary))
        if args.scope != 'cmm':
            assert sum(x['episodes'] for x in results) == 1754
            assert results[0]['summary']['frozen_model_before'] == results[1]['summary']['frozen_model_before'] == results[3]['summary']['frozen_model_before']
            native = dict(completed=True, source_commit=args.source_commit, phases=results,
                total_episodes=1754, total_native_actions=1754*80, learner_training=False)
            write(args.output_dir/'native-summary.json', native)
            if args.scope == 'native':
                write(args.output_dir/'summary.json', dict(native, tests=tests, calibration_launched=False))
                return
        if not tests['core_cpu']['passed']:
            raise RuntimeError('Native motor phases completed; recurrent-core CPU qualification failed; no CMM training')
        cuda_environment = dict(os.environ, FLUFFY_META_RL_CUDA_QUALIFICATION='1')
        tests['core_cuda'] = run_tests(root, args.output_dir, 'cuda-core-tests',
            ('tests/test_meta_rl_cuda.py',), environment=cuda_environment)
        write(args.output_dir/'tests.json', tests)
        if not tests['core_cuda']['passed']:
            raise RuntimeError('Native motor phases completed; recurrent-core CUDA qualification failed; no CMM training')
        write(args.output_dir/'progress.json', dict(phase='cmm-one-epoch-cost-smoke', completed_phases=4, total_phases=4))
        with (args.output_dir/'cmm-one-epoch-cost-smoke.log').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'fluffy_bassoon.meta_rl.fewshot_regression',
                '--source-commit', args.source_commit, '--output-dir', str(args.output_dir/'cmm-one-epoch-cost-smoke'),
                '--variant', 'cmm', '--shots', '20', '--seed', '0', '--epochs', '1',
                '--max-elapsed-seconds', '600', '--device', 'cuda'], cwd=root,
                stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
        calibration = read(args.output_dir/'cmm-one-epoch-cost-smoke/summary.json')
        if not calibration['completed'] or calibration['iteration'] != 157:
            raise RuntimeError('Bounded CMM cost smoke did not finish its one epoch; preserve checkpoint')
        write(args.output_dir/'summary.json', dict(completed=True, source_commit=args.source_commit,
            phases=results, total_episodes=sum(x['episodes'] for x in results),
            total_native_actions=sum(x['native_actions'] for x in results), learner_training=False,
            tests=tests, calibration_cost_smoke=calibration, full_calibration_launched=False))
    except BaseException as error:
        write(args.output_dir/'failure.json', dict(error_type=type(error).__name__, error=str(error),
            source_commit=args.source_commit, learner_training=False))
        raise


if __name__ == '__main__':
    main()
