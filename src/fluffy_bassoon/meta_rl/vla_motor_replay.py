"""Zero-interaction replay of fixed OpenPI motor inputs across runtime states.

This diagnostic reads completed robot traces. It never opens a simulator,
executes robot actions, restores a controller or updates any model weights.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_features import FrozenOpenPIPrefixFeatures, difference_metrics
from .vla_libero import FrozenOpenPI, OfficialSAC, OPENPI_COMMIT, sha256, write_json
from .vla_source40_corpus import PI05_RECEIPT
from .vla_source40_schedule import content_hash


STAGES = ('plain', 'prefix_attached', 'prefix_qualified', 'sac_initialized')
KEY_TYPES = ('typed', 'legacy')


def array_hash(value):
    value = np.asarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def checked_json(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'JSON checksum differs: {path}')
    value = json.loads(Path(path).read_text())
    if sha256(path) != expected:
        raise ValueError(f'JSON changed while reading: {path}')
    return value


def _trace(directory, receipt, *, wrapped):
    directory = Path(directory).resolve()
    path = (directory / receipt['trace']).resolve()
    if not path.is_relative_to(directory) or sha256(path) != receipt['trace_sha256']:
        raise ValueError('Trace path or checksum differs')
    with gzip.open(path, 'rb') as stream:
        data = pickle.load(stream)
    if sha256(path) != receipt['trace_sha256']:
        raise ValueError('Trace changed while reading')
    rows = data['rows'] if wrapped else data
    if not rows or len(rows) != receipt['chunks'] or not rows[-1]['done']:
        raise ValueError('Require the actual complete recorded episode')
    return rows[0]


def load_inputs(path, expected):
    manifest = checked_json(path, expected)
    if (manifest.get('format') != 'openpi_motor_replay_inputs_v1' or len(manifest['cases']) != 2
            or [row['task_id'] for row in manifest['cases']] != [0, 1]):
        raise ValueError('Require both predeclared task0/1 input comparisons')
    inputs = []
    for row in manifest['cases']:
        old = checked_json(row['reference_index'], row['reference_index_sha256'])
        new = checked_json(row['dsrl_index'], row['dsrl_index_sha256'])
        old_directory, new_directory = Path(row['reference_index']).parent, Path(row['dsrl_index']).parent
        old_contract = checked_json(old_directory / 'contract.json', row['reference_contract_sha256'])
        new_contract = checked_json(new_directory / 'contract.json', row['dsrl_contract_sha256'])
        expected_case = dict(benchmark='libero90', case_id=f'libero90:{row["task_id"]:02d}/state-0/seed-123',
            suite='libero_90', task_id=row['task_id'], initial_state=0, scene_seed=123)
        if (old.get('completed') is not True or new.get('completed') is not True
                or old.get('case') != expected_case or new.get('case') != expected_case
                or old.get('weights_unchanged') is not True or new.get('target_only') is not True
                or old_contract.get('motor_receipt_sha256') != PI05_RECEIPT
                or new_contract.get('motor_receipt_sha256') != PI05_RECEIPT):
            raise ValueError('Require completed native and target-only DSRL cases with the same official motor')
        old_receipts = [value for value in old['episodes'] if value['method'] == 'native'
                        and value['support_budget'] == 0 and value['role'] == 'query']
        new_receipts = [value for value in new['query_receipts'] if value['milestone'] == 0]
        if len(old_receipts) != 1 or len(new_receipts) != 1:
            raise ValueError('Require one exact initial native query from each run')
        old_receipt, new_receipt = old_receipts[0], new_receipts[0]
        if (new_receipt.get('completed') is not True or old_receipt.get('completed') is not True
                or new_receipt.get('query_behavior') != 'native' or new_receipt.get('target_optimizer_updates') != 0
                or not old_receipt.get('initial_state_vector_sha256')
                or old_receipt['initial_state_vector_sha256'] != new_receipt['initial_state_vector_sha256']
                or old_receipt['episode_random_seed'] != new_receipt['episode_random_seed']):
            raise ValueError('Native queries differ in case, randomness or learning stage')
        instruction = old_contract['simulator']['instruction']
        if not instruction or new['training_receipts'][0]['instruction'] != instruction:
            raise ValueError('Recorded motor instruction differs')
        first = _trace(old_directory, old_receipt, wrapped=False)
        second = _trace(new_directory, new_receipt, wrapped=True)
        for name in ('image', 'wrist', 'state'):
            if (np.asarray(first['motor_observation'][name]).dtype != np.asarray(second['motor_observation'][name]).dtype
                    or not np.array_equal(first['motor_observation'][name], second['motor_observation'][name])):
                raise ValueError(f'Cannot isolate runtime effects: recorded initial {name} differs')
        noise = np.asarray(first['native_initial_noise'])
        if (noise.shape != (10, 32) or noise.dtype != np.float32 or not np.isfinite(noise).all()
                or not np.array_equal(noise, second['native_initial_noise'])):
            raise ValueError('Cannot isolate runtime effects: actual initial H10 noise differs')
        for value in (first, second):
            if (np.asarray(value['executed_actions']).shape != (10, 7)
                    or not np.isfinite(value['executed_actions']).all()):
                raise ValueError('Require the actual complete first executed H10 action chunk')
        inputs.append(dict(task_id=row['task_id'], case=expected_case, instruction=instruction,
            query_seed=old_receipt['episode_random_seed'], noise=noise.copy(),
            reference_observation=first['motor_observation'], dsrl_observation=second['motor_observation'],
            reference_actions=np.asarray(first['executed_actions']).copy(),
            dsrl_actions=np.asarray(second['executed_actions']).copy(),
            input_sha256={name: array_hash(first['motor_observation'][name]) for name in ('image', 'wrist', 'state')},
            noise_sha256=array_hash(noise), reference_receipt=old_receipt, dsrl_receipt=new_receipt))
    return inputs, manifest


def parameter_signature(model):
    """Hash frozen NNX leaves sequentially, avoiding a full host model copy."""
    from flax import nnx
    import jax
    digest, dtypes, count = hashlib.sha256(), Counter(), 0
    for path, variable in sorted(nnx.state(model).flat_state().items(), key=lambda item: str(item[0])):
        value = getattr(variable, 'value', variable)
        array = np.asarray(jax.device_get(value))
        if array.dtype.hasobject:
            raise ValueError('Unexpected nonnumeric frozen model state')
        digest.update(str((path, array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
        dtypes[str(array.dtype)] += int(array.size)
        count += 1
    return dict(sha256=digest.hexdigest(), leaves=count, elements_by_dtype=dict(dtypes))


def runtime_configuration():
    import jax
    values = {name: value if isinstance(value, (str, bool, int, float, type(None))) else str(value)
        for name, value in jax.config.values.items()}
    versions = {}
    for name in ('jax', 'jaxlib', 'flax', 'numpy', 'torch', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(versions=versions, python=sys.version, executable=sys.executable,
        jax_config=values, jax_devices=[str(value) for value in jax.devices()],
        environment={name: os.environ.get(name) for name in ('XLA_FLAGS', 'CUDA_VISIBLE_DEVICES',
            'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_ENABLE_X64', 'JAX_DEFAULT_PRNG_IMPL',
            'JAX_COMPILATION_CACHE_DIR', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'NVIDIA_TF32_OVERRIDE')})
    try:
        from jax.lib import xla_bridge
        result['platform_version'] = xla_bridge.get_backend().platform_version
    except Exception as exc:
        result['platform_version_unavailable'] = type(exc).__name__
    probe = os.environ.get('LYCO_BAIGE_GPU_MAP')
    if probe:
        result['allocation_probe_sha256'] = sha256(probe)
        result['allocation_probe'] = json.loads(Path(probe).read_text())
    return result


def key(seed, kind):
    import jax
    return jax.random.key(seed) if kind == 'typed' else jax.random.PRNGKey(seed)


def lowered_fingerprint(policy, observation, instruction, noise, seed):
    """Read lowered HLO for the original captured module_jit, without rebinding it."""
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    from .vla_features import motor_element
    captured = inspect.getclosurevars(policy._sample_actions).nonlocals
    if not {'jitted_fn', 'state'}.issubset(captured):
        return dict(available=False, reason='qualified module_jit closure schema differs')
    prepared = policy._input_transform(jax.tree.map(lambda value: value, motor_element(observation, instruction)))
    prepared = Observation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[None], prepared))
    result = {}
    for kind in KEY_TYPES:
        _, sample_key = jax.random.split(key(seed, kind))
        lowered = captured['jitted_fn'].lower(captured['state'], sample_key, prepared,
            **dict(policy._sample_kwargs, noise=jnp.asarray(noise)[None]))
        hlo = lowered.compiler_ir(dialect='hlo').as_hlo_text()
        result[kind] = dict(hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(), hlo_text_bytes=len(hlo),
            key_dtype=str(sample_key.dtype), key_shape=list(sample_key.shape),
            key_data=np.asarray(jax.random.key_data(sample_key)).tolist())
    return dict(available=True, key_data_equal=result['typed']['key_data'] == result['legacy']['key_data'],
        lowered_hlo_text_equal=result['typed']['hlo_sha256'] == result['legacy']['hlo_sha256'], signatures=result)


def action_differences(reference, candidate):
    return dict(horizon=difference_metrics(reference, candidate),
        first_primitive=difference_metrics(np.asarray(reference)[0], np.asarray(candidate)[0]))


def replay_stage(motor, inputs, stage, *, deadline, key_order=KEY_TYPES):
    arrays, rows = {}, []
    for item in inputs:
        for origin in ('reference', 'dsrl'):
            observation = item[origin + '_observation']
            for kind in key_order:
                for repeat in range(2):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Motor replay elapsed-time ceiling reached')
                    motor.policy._rng = key(item['query_seed'], kind)
                    actions = np.asarray(motor.infer(observation, item['instruction'], item['noise']), np.float32)
                    label = f'task{item["task_id"]}_{origin}_{kind}_repeat{repeat}'
                    arrays[label] = actions
                    rows.append(dict(label=label, stage=stage, task_id=item['task_id'], origin=origin,
                        key_type=kind, repeat=repeat, output_sha256=array_hash(actions),
                        against_recorded_reference=action_differences(item['reference_actions'], actions),
                        against_recorded_dsrl=action_differences(item['dsrl_actions'], actions)))
    comparisons = []
    for item in inputs:
        task = f'task{item["task_id"]}'
        for origin in ('reference', 'dsrl'):
            for kind in KEY_TYPES:
                name = f'{task}_{origin}_{kind}'
                comparisons.append(dict(kind='same_key_repeat', name=name,
                    delta=action_differences(arrays[name + '_repeat0'], arrays[name + '_repeat1'])))
            comparisons.append(dict(kind='typed_vs_legacy', name=f'{task}_{origin}',
                delta=action_differences(arrays[f'{task}_{origin}_typed_repeat0'], arrays[f'{task}_{origin}_legacy_repeat0'])))
        for kind in KEY_TYPES:
            comparisons.append(dict(kind='identical_inputs_from_both_runs', name=f'{task}_{kind}',
                delta=action_differences(arrays[f'{task}_reference_{kind}_repeat0'], arrays[f'{task}_dsrl_{kind}_repeat0'])))
    return dict(stage=stage, predictions=rows, comparisons=comparisons, runtime=runtime_configuration()), arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'output', 'checkpoint', 'openpi-root', 'dsrl-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('inputs-sha256', 'source-commit', 'expected-hostname'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-elapsed-seconds', type=float, default=480.)
    parser.add_argument('--key-order', choices=('typed-first', 'legacy-first'), default='typed-first')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname or not 0 < args.max_elapsed_seconds <= 480:
        raise ValueError('Require the explicit allocated Linux host and <=480s zero-action diagnostic')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed diagnostic source')
    ready = json.loads((args.checkpoint / 'READY.json').read_text())
    if (sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT or ready.get('status') != 'verified'
            or ready.get('checkpoint') != 'gs://openpi-assets/checkpoints/pi05_libero'
            or any((args.checkpoint / row['name']).stat().st_size != row['bytes'] for row in ready['files'])
            or sha256(args.checkpoint / 'assets/physical-intelligence/libero/norm_stats.json')
            != 'b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'):
        raise ValueError('Official pi0.5 checkpoint readiness, receipt, file sizes or normalization differs')
    inputs, manifest = load_inputs(args.inputs, args.inputs_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    deadline, started = time.monotonic() + args.max_elapsed_seconds, time.monotonic()
    before_runtime = runtime_configuration()
    args.model_config, args.feature_interface = 'pi05_libero', 'pixel-sac'
    motor = FrozenOpenPI(args)
    before_parameters = parameter_signature(motor.policy._model)
    sample_callable = motor.policy._sample_actions
    outputs, stage_rows, feature_qualification, sac = {}, [], [], None
    key_order = KEY_TYPES if args.key_order == 'typed-first' else tuple(reversed(KEY_TYPES))
    for stage in STAGES:
        if stage == 'prefix_attached':
            motor.feature_encoder = FrozenOpenPIPrefixFeatures(motor.policy,
                checkpoint_receipt_sha256=PI05_RECEIPT, openpi_commit=OPENPI_COMMIT, model_config='pi05_libero')
        elif stage == 'prefix_qualified':
            for item in inputs:
                feature_qualification.append(dict(task_id=item['task_id'],
                    result=motor.feature_encoder.qualify(item['reference_observation'], item['instruction'])))
        elif stage == 'sac_initialized':
            sac = OfficialSAC(args.dsrl_root, 0)
            if sac.updates != 0 or sac.replay:
                raise RuntimeError('Diagnostic SAC must be fresh with zero updates and replay')
        if motor.policy._sample_actions is not sample_callable:
            raise RuntimeError('A diagnostic stage replaced the canonical motor sample callable')
        result, arrays = replay_stage(motor, inputs, stage, deadline=deadline, key_order=key_order)
        if stage == 'plain':
            try:
                result['lowered_motor'] = lowered_fingerprint(motor.policy, inputs[0]['reference_observation'],
                    inputs[0]['instruction'], inputs[0]['noise'], inputs[0]['query_seed'])
            except Exception as exc:
                result['lowered_motor'] = dict(available=False, error=type(exc).__name__ + ': ' + str(exc))
        result['against_plain_stage'] = [] if stage == 'plain' else [dict(label=name,
            delta=action_differences(outputs['plain'][name], value)) for name, value in arrays.items()]
        path = args.output / f'{stage}-actions.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        result.update(actions_path=path.name, actions_sha256=sha256(path))
        outputs[stage], stage_rows = arrays, [*stage_rows, result]
        write_json(args.output / 'progress.json', dict(completed_stages=stage_rows, new_simulator_actions=0))
    after_parameters = parameter_signature(motor.policy._model)
    if before_parameters != after_parameters:
        raise RuntimeError('Frozen motor parameter state changed during zero-action replay')
    source_files = [args.openpi_root / name for name in ('src/openpi/policies/policy.py',
        'src/openpi/models/pi0.py', 'src/openpi/shared/nnx_utils.py', 'src/openpi/policies/policy_config.py')]
    provenance = dict(inputs_manifest_path=str(args.inputs), inputs_manifest_sha256=args.inputs_sha256, inputs_manifest=manifest,
        source_commit=args.source_commit, motor=motor.metadata, before_runtime=before_runtime,
        parameters_before=before_parameters, parameters_after=after_parameters,
        source_files={str(path): sha256(path) for path in source_files}, key_order=list(key_order),
        feature_qualification=feature_qualification, sac_initialized_only=sac is not None,
        fixed_inputs=[dict(task_id=item['task_id'], case=item['case'], instruction=item['instruction'],
            query_seed=item['query_seed'], observation_sha256=item['input_sha256'], noise_sha256=item['noise_sha256'],
            recorded_reference_delta=action_differences(item['reference_actions'], item['dsrl_actions'])) for item in inputs])
    write_json(args.output / 'index.json', dict(format='openpi_zero_action_motor_replay_v1', completed=True,
        elapsed_seconds=time.monotonic() - started, provenance=provenance, stages=stage_rows,
        motor_parameters_unchanged=True, canonical_sample_callable_unchanged=True,
        simulator_started=False, new_simulator_actions=0, optimizer_updates=0,
        interpretation='Numerical runtime replay only. Different compiled signatures or observed prediction errors do not establish a robot performance ranking.'))
    print(json.dumps(dict(completed=True, output=str(args.output), new_simulator_actions=0,
        elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
