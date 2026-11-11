#!/usr/bin/env bash
# CPU preparation only; separate from the qualified simulator environment.
set -euo pipefail
TASK_PROJECT=${1:?project root}
TASK_CODE=${2:?verified checkout}
TASK_SETUP="$TASK_PROJECT/setups/libero-procedural-pi05-20261111-r1"
TASK_UV="$TASK_PROJECT/setups/libero-gen-history-pilot-20261111-r1/uv-0.13.7/bin/uv"
TASK_FILES="$TASK_CODE/outputs/libero-procedural-qual-20261111-r1"
TASK_SOURCE="$TASK_SETUP/openpi"
mkdir -p "$TASK_SETUP"
export UV_CACHE_DIR="$TASK_SETUP/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_SETUP/python"
export UV_PROJECT_ENVIRONMENT="$TASK_SETUP/env"
export HF_HOME="$TASK_SETUP/huggingface-cache"
export HF_HUB_DISABLE_XET=1
export GIT_LFS_SKIP_SMUDGE=1
python - "$TASK_SETUP" <<'PY'
import json,pathlib,shutil,subprocess,sys
p=pathlib.Path(sys.argv[1]); disk=shutil.disk_usage(p)
quota=shutil.which('quota')
record={'disk':dict(zip(('total','used','free'),disk)), 'quota_command':quota}
if quota:
 r=subprocess.run([quota,'-s'],capture_output=True,text=True,timeout=30)
 record.update(quota_returncode=r.returncode,quota_output=r.stdout+r.stderr)
(p/'write-probe').write_text('ok'); assert (p/'write-probe').read_text()=='ok'
(p/'storage.json').write_text(json.dumps(record,indent=2)+'\n')
PY
if [[ ! -d "$TASK_SOURCE/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/austinapatel/openpi.git "$TASK_SOURCE"
  git -C "$TASK_SOURCE" checkout --detach 4066d8f7c544ba1e5cd5386b2db47ab900fae3fc
fi
test "$(git -C "$TASK_SOURCE" rev-parse HEAD)" = 4066d8f7c544ba1e5cd5386b2db47ab900fae3fc
git -C "$TASK_SOURCE" diff --quiet HEAD
"$TASK_UV" sync --project "$TASK_SOURCE" --python 3.11 --frozen --no-dev
"$TASK_UV" pip freeze --python "$TASK_SETUP/env/bin/python" > "$TASK_SETUP/installed.txt"
# Add only the official client's transport dependencies to a separate directory.
# The existing Python 3.10 simulator and its numerical libraries are untouched.
"$TASK_UV" pip install --python "$TASK_PROJECT/setups/libero-gen-history-pilot-20261111-r1/env/bin/python" \
  --target "$TASK_SETUP/client-deps" --no-deps 'websockets==15.0.1' 'msgpack==1.1.0' 'typing-extensions==4.15.0'
"$TASK_SETUP/env/bin/python" - "$TASK_FILES/SOURCE.json" "$TASK_SETUP" <<'PY'
import hashlib,json,pathlib,sys
from huggingface_hub import snapshot_download
source=json.loads(pathlib.Path(sys.argv[1]).read_text()); setup=pathlib.Path(sys.argv[2])
root=pathlib.Path(snapshot_download(source['checkpoint'],revision=source['revision'],
    allow_patterns=[x['rfilename'] for x in source['inference_files']],
    local_dir=setup/'checkpoint',token=False,max_workers=4))
checked=[]
for item in source['inference_files']:
 p=root/item['rfilename']; assert p.stat().st_size==item['size'],str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''): h.update(block)
 value=h.hexdigest()
 if 'lfs' in item: assert value==item['lfs']['sha256'],str(p)
 checked.append({'file':item['rfilename'],'sha256':value,'bytes':p.stat().st_size})
(setup/'checkpoint-verified.json').write_text(json.dumps({'revision':source['revision'],'files':checked},indent=2)+'\n')
print('PI05_CPU_PREPARATION_COMPLETE',flush=True)
PY
