#!/usr/bin/env bash
set -euo pipefail
root="${REGROUND_RUNTIME_ROOT:-/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1}"
setup_dir=$(cd -- "$(dirname -- "$0")" && pwd)
uv="${REGROUND_UV:-/mowenjie/fluffy-bassoon-2/tools/uv-20261028/bin/uv}"
base="${REGROUND_PYTHON:-/mowenjie/.uv-python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12}"
upstream="$root/upstream"
rev=16a98486f560676b9036d1a6d067b9dd51a5917a
if [[ ! -d "$upstream" ]]; then
    git clone https://github.com/umersiddique94/ReGround.git "$upstream"
    git -C "$upstream" checkout --detach "$rev"
fi
[[ $(git -C "$upstream" rev-parse HEAD) == "$rev" ]]
git -C "$upstream" diff --quiet "$rev" -- .
[[ $("$uv" --version) == "uv 0.9.5" ]]
export UV_CACHE_DIR="$root/uv-cache"
if [[ ! -x "$root/env/bin/python" ]]; then
    "$uv" venv --python "$base" "$root/env"
fi
py="$root/env/bin/python"
"$py" -c 'import sys; assert sys.version_info[:3] == (3,12,15) and sys.prefix != sys.base_prefix'
mirror=()
if [[ -n ${REGROUND_PYPI_MIRROR:-} ]]; then mirror=(--pypi-mirror "$REGROUND_PYPI_MIRROR"); fi
"$py" "$setup_dir/fetch_packages.py" "$root/wheels" "${mirror[@]}"
"$uv" pip install --python "$py" --no-index --find-links "$root/wheels" \
    -r "$upstream/requirements.txt" -r "$setup_dir/requirements-linux.txt"
"$uv" pip check --python "$py"
"$py" "$setup_dir/prepare_cache.py" \
    --model "${REGROUND_MODEL_SOURCE:-/mowenjie/fluffy-bassoon/inputs/pi05-libero-lerobot-v044-7152f13e-20261027-r1}" \
    --tokenizer "${REGROUND_TOKENIZER_SOURCE:-/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}" \
    --hub "$root/hf/hub" > "$root/hf-verified.jsonl"
"$py" "$setup_dir/prepare_assets.py" "${REGROUND_ASSET_SOURCE:-/mowenjie/fluffy-bassoon/inputs/libero-assets-20261107}" \
    "$root/assets" > "$root/assets-verified.json"
assetdir="$root/assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
pkg=$("$py" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("libero").submodule_search_locations)))')
dest="$pkg/libero/assets"
if [[ ! -e "$dest" ]]; then ln -s "$assetdir" "$dest"; fi
[[ $(realpath "$dest") == "$assetdir" ]]
export REGROUND_RUNTIME_ROOT="$root"
source "$setup_dir/activate.sh"
"$py" "$upstream/main.py" setup
