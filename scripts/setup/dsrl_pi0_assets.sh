#!/usr/bin/env bash
# New allocated CPU identity; reuse the ended, imported R4 runtime read-only.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed code}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_ID=dsrl-pi0-task57-20261101-prep-r7
TASK_RUNTIME=dsrl-pi0-task57-20261101-prep-r4
TASK_CODE=$(pwd)
TASK_OUT=$TASK_ROOT/outputs/$TASK_ID
TASK_REPO=$TASK_ROOT/deps/$TASK_RUNTIME/repo
TASK_PY=$TASK_ROOT/envs/$TASK_RUNTIME/bin/python
TASK_UV=$TASK_ROOT/deps/$TASK_RUNTIME/tools/bin/uv
TASK_STARTED=$(date +%s)
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_OUT/exit-code.txt"; printf "%s\n" "$(( $(date +%s) - TASK_STARTED ))" > "$TASK_OUT/elapsed-seconds.txt"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$TASK_CODE/src:$TASK_REPO
export DSRL_OFFICIAL_ROOT=$TASK_REPO
export OPENPI_DATA_HOME=$TASK_ROOT/cache/$TASK_ID
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/$TASK_RUNTIME/libero-config
export XDG_CACHE_HOME=$TASK_ROOT/cache/$TASK_ID/xdg
export JAX_COMPILATION_CACHE_DIR=$TASK_ROOT/cache/$TASK_ID/jax-cpu
"$TASK_PY" - "$TASK_OUT" "$TASK_SOURCE" <<'PY'
import glob,json,pathlib,socket,sys
out,source=pathlib.Path(sys.argv[1]),sys.argv[2]
assert sys.platform=='linux' and socket.gethostname().startswith('job-')
assert not glob.glob('/dev/nvidia[0-9]*'), 'Expected an actual0GPU allocation'
value=dict(hostname=socket.gethostname(),source_commit=source,physical_gpu_devices=[],
    cpu_max=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text(),
    memory_max=pathlib.Path('/sys/fs/cgroup/memory.max').read_text())
(out/'allocation.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
PY
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
if ! "$TASK_PY" -m pytest -q -p no:cacheprovider \
  --junitxml="$TASK_OUT/qualification.xml" tests/test_dsrl_pi0_replication.py \
  > "$TASK_OUT/qualification.log" 2>&1; then
  cat "$TASK_OUT/qualification.log"
  exit 1
fi
tail -n 2 "$TASK_OUT/qualification.log"
"$TASK_PY" - "$TASK_OUT" "$TASK_CODE" "$TASK_SOURCE" <<'PY'
import hashlib,json,pathlib,sys
out,code,source=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
paths=['src/fluffy_bassoon/meta_rl/dsrl_pi0_replication.py','tests/test_dsrl_pi0_replication.py']
value=dict(source_commit=source,passed=True,exit_code=0,full_suite=True,
    file_sha256={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in paths})
(out/'qualification-summary.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
PY
"$TASK_PY" - "$TASK_OUT" "$TASK_REPO" <<'PY'
import base64,hashlib,importlib.metadata as md,json,pathlib,sys
import fsspec
out,repo=map(pathlib.Path,sys.argv[1:])
from openpi.training import config
from openpi.shared import download
conf=config.get_config('pi0_libero')
assert conf.model.action_horizon==50 and conf.model.action_dim==32
url='gs://openpi-assets/checkpoints/pi0_libero'
fs=fsspec.filesystem('gcs',token='anon')
remote=fs.find('openpi-assets/checkpoints/pi0_libero',detail=True)
if not remote: raise FileNotFoundError('Official canonical Pi0 LIBERO checkpoint is unavailable')
(out/'remote-objects.json').write_text(json.dumps(remote,indent=2,default=str)+'\n')
print(json.dumps(dict(stage='official_gcs_listing',objects=len(remote),
    bytes=sum(int(row.get('size',0)) for row in remote.values()))),flush=True)
checkpoint=download.maybe_download(url,token='anon')
files=[]
for path in sorted(checkpoint.rglob('*')):
    if not path.is_file(): continue
    sha,md5=hashlib.sha256(),hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            sha.update(block);md5.update(block)
    name=str(path.relative_to(checkpoint))
    metadata=remote['openpi-assets/checkpoints/pi0_libero/'+name]
    expected=metadata.get('md5Hash')
    actual=base64.b64encode(md5.digest()).decode()
    if expected is not None and expected!=actual: raise ValueError('GCS MD5 differs: '+name)
    if int(metadata['size'])!=path.stat().st_size: raise ValueError('GCS file size differs: '+name)
    files.append(dict(path=name,bytes=path.stat().st_size,sha256=sha.hexdigest(),
                      gcs_md5=expected,md5_matches=expected==actual if expected else None))
receipt=dict(status='verified',checkpoint=url,files=files,bytes=sum(row['bytes'] for row in files),
    original_logical_url='s3://openpi-assets/checkpoints/pi0_libero',
    legacy_s3_byte_equivalence='not independently established: legacy S3 bucket returns NoSuchBucket',
    model_config='pi0_libero',model_loaded=False,requested_model_family_unchanged=True)
(out/'checkpoint-path.txt').write_text(str(checkpoint)+'\n')
(out/'checkpoint-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
from openpi.models.tokenizer import PaligemmaTokenizer
PaligemmaTokenizer(conf.model.max_token_len)
versions={name:md.version(name) for name in ('jax','jaxlib','flax','numpy','pillow','mujoco','robosuite','torch','tensorflow','orbax-checkpoint')}
(out/'runtime.json').write_text(json.dumps(dict(versions=versions,python=sys.version,
    runtime_reused_read_only=str(pathlib.Path(sys.prefix)),model_config='pi0_libero',
    native_horizon=50,action_dim=32,model_loaded=False,simulator_constructed=False),indent=2)+'\n')
print(json.dumps(dict(stage='checkpoint_verified',files=len(files),bytes=receipt['bytes'],versions=versions)),flush=True)
PY
"$TASK_UV" pip freeze --python "$TASK_PY" > "$TASK_OUT/requirements-frozen.txt"
