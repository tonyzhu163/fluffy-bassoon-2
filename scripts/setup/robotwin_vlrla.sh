#!/usr/bin/env bash
# Create an isolated overlay; never edit the shared runtime or existing runs.
set -euo pipefail
if [[ $# != 7 ]]; then
  echo "usage: $0 UV BASE_PYTHON ENV_DIR OUTPUT_DIR RLINF_ROOT ROBOTWIN_ROOT MODEL_DIR" >&2
  exit 2
fi
TASK_UV=$1
TASK_BASE_PYTHON=$2
TASK_ENV=$3
TASK_OUTPUT=$4
TASK_RLINF=$5
TASK_ROBOTWIN=$6
TASK_MODEL=$7
TASK_PROJECT=$(cd "$(dirname "$0")/../.." && pwd)
[[ $(uname -s) == Linux ]]
[[ ! -e $TASK_ENV && ! -e $TASK_OUTPUT ]]
[[ -x $TASK_UV && -x $TASK_BASE_PYTHON ]]
[[ $(git -C "$TASK_RLINF" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$TASK_ROBOTWIN" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$TASK_RLINF" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$TASK_ROBOTWIN" status --porcelain --untracked-files=no) ]]
mkdir -p "$(dirname "$TASK_ENV")" "$TASK_OUTPUT"
df -h "$(dirname "$TASK_ENV")" > "$TASK_OUTPUT/storage.txt"
TASK_WRITE_PROBE=$(mktemp "$(dirname "$TASK_ENV")/.robotwin-write-probe.XXXXXX")
printf 'write probe\n' > "$TASK_WRITE_PROBE"
rm "$TASK_WRITE_PROBE"
"$TASK_BASE_PYTHON" -c 'import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ["torch","numpy","sapien","mplib","transformers","rlinf-openpi"]},sort_keys=True))' > "$TASK_OUTPUT/shared-before.json"
"$TASK_UV" venv --python "$TASK_BASE_PYTHON" "$TASK_ENV"
TASK_BASE_SITE=$("$TASK_BASE_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
TASK_LOCAL_SITE=$("$TASK_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$TASK_BASE_SITE" > "$TASK_LOCAL_SITE/shared_runtime.pth"
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" 'mplib==0.2.1' 'open3d==0.19.0' 'numpy==1.26.4'
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export PYTHONPATH="$TASK_RLINF:$TASK_ROBOTWIN:$TASK_PROJECT/src"
export PYTHONDONTWRITEBYTECODE=1
"$TASK_ENV/bin/python" "$TASK_PROJECT/scripts/diagnostics/robotwin_vlrla_inventory.py" \
  --robotwin-root "$TASK_ROBOTWIN" --model-path "$TASK_MODEL" --output "$TASK_OUTPUT/setup.json"
"$TASK_BASE_PYTHON" -c 'import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ["torch","numpy","sapien","mplib","transformers","rlinf-openpi"]},sort_keys=True))' > "$TASK_OUTPUT/shared-after.json"
cmp "$TASK_OUTPUT/shared-before.json" "$TASK_OUTPUT/shared-after.json"
