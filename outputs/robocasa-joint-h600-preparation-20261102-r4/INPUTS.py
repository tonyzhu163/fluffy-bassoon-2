"""Run-local immutable input preparation; CPU-only, before model startup."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RLINF = ROOT / 'sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf'
RUN_ID = 'robocasa-joint-h600-preparation-20261102-r4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(code):
    base = copy.deepcopy(read(code / 'outputs/robocasa-joint-six-update-proposal-20261102-r1/BINDING-TEMPLATE.json')['policy_profile'])
    migrations = []
    # Resolve repository receipts in this exact Git checkout, never through a
    # historical output-directory copy. Each original content hash is retained.
    for ref in [base['loading_qualification'], *base['core_qualification']]:
        original = Path(ref['path'])
        pinned = code / original.relative_to(ROOT)
        if pinned.is_file():
            assert sha(pinned) == ref['sha256'], str(pinned)
            ref['path'] = str(pinned)
            migrations.append(dict(original=str(original), pinned=str(pinned), sha256=ref['sha256']))
        assert Path(ref['path']).is_file() and sha(ref['path']) == ref['sha256'], ref['path']
    loading = read(base['loading_qualification']['path'])
    assert loading['completed'] and loading['converted_norm_identical'] and not loading['has_value_head']
    assert len(loading['actual_loader']) == 1
    assert not loading['actual_loader'][0]['missing_keys'] and not loading['actual_loader'][0]['unexpected_keys']
    initializer = base['initializer']
    assert sha(initializer['norm_path']) == initializer['normalization_sha256'] == loading['normalization_sha256']
    flora = read(code / 'outputs/robocasa-flora-throughput-20261102-r2/INPUTS.json')
    assert sha(flora['model_receipt']['path']) == flora['model_receipt']['sha256']
    native_env = dict(
        PYTHONPATH=str(code / 'src') + ':' + str(RLINF), RLINF_PATH=str(RLINF),
        ROBOCASA_ASSETS_PATH=str(ROOT / 'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(ROOT / 'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        ROBOCASA_PI05_TORCH_MODEL=initializer['model_directory'],
        ROBOCASA_PI05_NATIVE_NORM=initializer['norm_path'])
    warmup = code / 'outputs' / RUN_ID / 'warmup-episodes.json'
    assert len(read(warmup)) == 8
    return base, native_env, flora, warmup, migrations


def check(code, *, full):
    base, native_env, flora, warmup, migrations = load_inputs(code)
    head = lambda path: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    assert head(RLINF) == base['rlinf_execution_commit']
    assert head(flora['official_code']) == flora['official_code_commit']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
    files = []
    def inspect(path, expected=None, *, hash_file=True):
        path = Path(path); stat = path.stat(); assert path.is_file() and os.access(path, os.R_OK)
        row = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if hash_file:
            row['sha256'] = sha(path)
            if expected is not None: assert row['sha256'] == expected, str(path)
        files.append(row)
    initializer = base['initializer']
    inspect(Path(initializer['model_directory']) / 'model.safetensors', initializer['model_sha256'], hash_file=full)
    inspect(Path(initializer['model_directory']) / 'config.json')
    inspect(initializer['norm_path'], initializer['normalization_sha256'])
    for ref in [base['loading_qualification'], *base['core_qualification'], flora['model_receipt']]:
        inspect(ref['path'], ref['sha256'])
    model = read(flora['model_receipt']['path']); assert model['completed'] and model['model_directory'] == flora['model']
    for row in model['files']:
        assert Path(row['path']).stat().st_size == row['bytes']
        # The unchanged qualified Flora weights retain their download receipt;
        # rehash small config/tokenizer/code files and verify every shard exists.
        inspect(row['path'], row['sha256'], hash_file=row['bytes'] < 32 << 20)
    inspect(warmup)
    for episode in read(warmup):
        assert episode['horizon'] == episode['frame_clock']['physical_steps'] == 600
        assert episode['frame_clock']['frame_count'] == 601 and len(episode['videos']) == 3
        for ref in episode['videos']:
            inspect(ref['path'], ref['sha256'], hash_file=full)
    for relative in ('bin/nvcc', 'lib'):
        assert (Path(flora['toolkit']) / relative).exists()
    for key in ('ROBOCASA_ASSETS_PATH', 'OPENPI_DATA_HOME'):
        assert Path(native_env[key]).is_dir() and any(Path(native_env[key]).iterdir())
    for path in (ROOT / 'setups/robocasa365-native-20261101-r1/env/bin/python',
                 ROOT / 'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'):
        assert path.is_file() and os.access(path, os.X_OK)
    if not full:
        receipt = read(code / 'outputs' / RUN_ID / 'INPUT-CLOSURE.json')
        assert receipt['completed'] and receipt['full_native_model_hash_verified']
        for row in receipt['files']:
            stat = Path(row['path']).stat()
            assert stat.st_size == row['bytes'] and stat.st_mtime_ns == row['mtime_ns'], row['path']
    return dict(completed=True, source_commit=head(code), checked_unix=time.time(),
        full_native_model_hash_verified=full, model_computation=False, physical_actions=0,
        files=files, reference_migrations=migrations, native_env=native_env,
        flora_weight_validation='All11 qualified shards present with exact receipt sizes; no new full Flora weight rehash',
        future_generated_inputs='Five fresh gates are generated and verified before common-before; no old gates admitted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    result = check(args.code, full=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, checked_files=len(result['files']), model_computation=False)))
