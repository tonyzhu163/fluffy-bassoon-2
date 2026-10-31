#!/usr/bin/env bash
# Run only in an allocated Linux CPU job; never edits an existing runtime.
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed repository commit}
TASK_CODE=$(pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_ID=${2:?fresh immutable CPU preparation identity}
case "$TASK_ID" in dsrl-pi0-task57-20261101-prep-r[0-9]*) ;; *) exit 2;; esac
TASK_OUT=$TASK_ROOT/outputs/$TASK_ID
TASK_DEPS=$TASK_ROOT/deps/$TASK_ID
TASK_ENV=$TASK_ROOT/envs/$TASK_ID
TASK_STARTED=$(date +%s)
test "$(uname -s)" = Linux
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT" "$TASK_DEPS"
exec > >(tee "$TASK_OUT/setup.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_OUT/exit-code.txt"; printf "%s\n" "$(( $(date +%s) - TASK_STARTED ))" > "$TASK_OUT/elapsed-seconds.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_BASE=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
"$TASK_BASE" - "$TASK_OUT" <<'PY'
import json, os, pathlib, shutil, socket, subprocess, sys
out = pathlib.Path(sys.argv[1])
import torch
assert torch.cuda.device_count() == 0, 'Preparation requires a real zero-GPU allocation'
assert socket.gethostname().startswith('job-')
disk = shutil.disk_usage(out)
assert disk.free >= 100 * 1024**3, 'Reserve space for isolated runtime and original weights'
probe = out / 'write-probe'
probe.write_text('write verified\n')
probe.unlink()
record = dict(hostname=socket.gethostname(), gpu_count=0,
              disk_free_bytes=disk.free, write_probe=True,
              cpu_max=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text(),
              memory_max=pathlib.Path('/sys/fs/cgroup/memory.max').read_text())
(out/'allocation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
PY
df -h "$TASK_ROOT"
if command -v quota >/dev/null; then quota -s || true; fi
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export UV_CACHE_DIR=$TASK_ROOT/cache/uv-dsrl-pi0-official
export UV_PYTHON_INSTALL_DIR=$TASK_ROOT/envs/uv-python
if command -v uv >/dev/null; then TASK_UV=$(command -v uv)
elif [ -x /mowenjie/.local/bin/uv ]; then TASK_UV=/mowenjie/.local/bin/uv
else
  # The qualified training venv deliberately omits pip. Bootstrap from its
  # bundled ensurepip wheel into this new tools directory, never into that venv.
  "$TASK_BASE" - "$TASK_DEPS/tools" <<'PY'
import ensurepip, pathlib, runpy, sys
target = sys.argv[1]
wheel = next((pathlib.Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'))
sys.path.insert(0, str(wheel))
sys.argv = ['pip', 'install', '--index-url', 'https://pypi.org/simple',
            '--target', target, 'uv==0.6.14']
runpy.run_module('pip', run_name='__main__')
PY
  TASK_UV=$TASK_DEPS/tools/bin/uv
fi
TASK_REPO=$TASK_DEPS/repo
git clone --no-checkout https://github.com/nakamotoo/dsrl_pi0.git "$TASK_REPO"
git -C "$TASK_REPO" checkout --detach 7f48937d4553e95244cd81c79236a3256df80597
git -C "$TASK_REPO" -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
test "$(git -C "$TASK_REPO/openpi" rev-parse HEAD)" = a6d2400d2534ce32e7bdf8747709b97aaef8ec04
test "$(git -C "$TASK_REPO/LIBERO" rev-parse HEAD)" = 87edbd1b823e464bc66557d213cc45e9f88aa543
"$TASK_UV" python install 3.11.11
"$TASK_UV" venv --python 3.11.11 "$TASK_ENV"
TASK_PY=$TASK_ENV/bin/python
"$TASK_UV" pip install --python "$TASK_PY" torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# Upstream requirements pin Pillow10.4 while its OpenPI metadata asks >=11.
# Preserve the released DSRL image library and NumPy1 for Robosuite1.4.1.
printf 'pillow==10.4.0\nnumpy==1.26.4\ntorch==2.6.0+cpu\nmujoco==3.3.1\n' > "$TASK_OUT/overrides.txt"
"$TASK_UV" pip install --python "$TASK_PY" --override "$TASK_OUT/overrides.txt" \
  -r "$TASK_REPO/requirements.txt" -e "$TASK_REPO" \
  -e "$TASK_REPO/openpi/packages/openpi-client" -e "$TASK_REPO/openpi" -e "$TASK_REPO/LIBERO" pytest
export OPENPI_DATA_HOME=$TASK_ROOT/cache/$TASK_ID
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/$TASK_ID/libero-config
mkdir -p "$LIBERO_CONFIG_PATH"
"$TASK_PY" - "$TASK_OUT" "$TASK_REPO" <<'PY'
import importlib.metadata as md, json, pathlib, sys
import yaml
out, repo = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(repo))
config = pathlib.Path(__import__('os').environ['LIBERO_CONFIG_PATH'])
libero = repo/'LIBERO/libero/libero'
paths = dict(benchmark_root=str(libero), bddl_files=str(libero/'bddl_files'),
             init_states=str(libero/'init_files'), datasets=str(repo/'datasets'),
             assets=str(libero/'assets'))
(config/'config.yaml').write_text(yaml.safe_dump(paths))
from examples import train_utils_sim
from openpi.training import config as pc
conf = pc.get_config('pi0_libero')
assert conf.model.action_horizon == 50 and conf.model.action_dim == 32
versions = {name: md.version(name) for name in
    ('jax','jaxlib','flax','numpy','pillow','mujoco','robosuite','torch','tensorflow','orbax-checkpoint')}
(out/'runtime.json').write_text(json.dumps(dict(versions=versions,
    python=sys.version, model_config='pi0_libero', native_horizon=50, action_dim=32,
    model_loaded=False, simulator_constructed=False), indent=2)+'\n')
print(json.dumps(versions), flush=True)
from openpi.shared import download
checkpoint = download.maybe_download('s3://openpi-assets/checkpoints/pi0_libero')
(out/'checkpoint-path.txt').write_text(str(checkpoint)+'\n')
import hashlib
files=[]
for path in sorted(checkpoint.rglob('*')):
    if not path.is_file(): continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):digest.update(block)
    files.append(dict(path=str(path.relative_to(checkpoint)), bytes=path.stat().st_size, sha256=digest.hexdigest()))
receipt=dict(status='verified', checkpoint='s3://openpi-assets/checkpoints/pi0_libero',
             files=files, bytes=sum(x['bytes'] for x in files), model_loaded=False)
(out/'checkpoint-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(checkpoint=str(checkpoint), files=len(files), bytes=receipt['bytes'])), flush=True)
PY
"$TASK_UV" pip freeze --python "$TASK_PY" > "$TASK_OUT/requirements-frozen.txt"
export DSRL_OFFICIAL_ROOT=$TASK_REPO
export PYTHONPATH=$TASK_CODE/src:$TASK_REPO
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$TASK_PY" -m pytest -q -p no:cacheprovider \
  --junitxml="$TASK_OUT/qualification.xml" "$TASK_CODE/tests/test_dsrl_pi0_replication.py" \
  > "$TASK_OUT/qualification.log" 2>&1
git -C "$TASK_REPO" status --porcelain > "$TASK_OUT/upstream-status.txt"
printf 'Preparation complete; no GPU/model/simulator run.\n'
