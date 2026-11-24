#!/usr/bin/env bash
# Run at the factory revision with an external scratch directory and an empty results directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cache=$(realpath -m "${1:?external scratch directory required}")
out=$(realpath -m "${2:?results directory required}")
test ! -e "$out/scores.json"
mkdir -p "$cache"/{policy,vlm,libero-config} "$out"

uv venv --python "${SIM_PYTHON:-3.10.12}" "$cache/sim"
uv venv --python "${POLICY_PYTHON:-3.12.16}" "$cache/env"
for env in sim env; do
    uv pip install --python "$cache/$env/bin/python" --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match -r "$here/requirements-$env.txt"
done
pin=8f1084e3132a39270c3a13ebe37270a43ece2a01
source_dir="$cache/LIBERO-$pin"
curl -fsSL --retry 2 "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/zip/$pin" -o "$cache/source.zip"
echo "53b6776952c61959f75ac9a9e413f0cd383ab7a9b98f24a899e50b61ec315fb6  $cache/source.zip" | sha256sum -c -
unzip -qo "$cache/source.zip" -d "$cache"
"$cache/sim/bin/python" - "$cache/libero-config/config.yaml" "$source_dir/libero/libero" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[2])
folders = dict(benchmark_root="", init_states="init_files", assets="assets", bddl_files="bddl_files", datasets=".")
Path(sys.argv[1]).write_text(json.dumps({k: str(root / v) for k,v in folders.items()}))
PY
base=https://huggingface.co/lerobot/smolvla_libero/resolve/a24f1f380461b665158d52acd140152c79e8b36d
for file in config.json model.safetensors policy_{pre,post}processor.json \
            policy_preprocessor_step_5_normalizer_processor.safetensors \
            policy_postprocessor_step_0_unnormalizer_processor.safetensors; do
    curl -fsSL --retry 2 "$base/$file" -o "$cache/policy/$file"
done
base=https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/4cde9d208e19519c36917aa170fffc7e381aa0ad
for file in config.json added_tokens.json chat_template.json generation_config.json merges.txt \
            preprocessor_config.json processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
    curl -fsSL --retry 2 "$base/$file" -o "$cache/vlm/$file"
done
(cd "$cache" && sha256sum -c "$here/downloads.sha256")
env -u CUDA_VISIBLE_DEVICES PYTHONHASHSEED=0 LP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    LIBGL_ALWAYS_SOFTWARE=1 LIBERO_CONFIG_PATH="$cache/libero-config" PYTHONPATH="$source_dir" \
    OPENBLAS_NUM_THREADS=1 "$cache/sim/bin/python" "$here/render.py" "$cache/frames"
cp "$cache/frames/frames.json" "$out/"
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 \
    "$cache/env/bin/python" "$here/evaluate.py" "$cache" "$out"
"$cache/env/bin/python" "$here/summarize.py" "$out"
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 \
    "$cache/env/bin/python" "$here/verify.py" "$cache" "$out"
