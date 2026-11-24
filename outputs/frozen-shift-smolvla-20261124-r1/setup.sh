#!/usr/bin/env bash
# Prepare SCRATCH with the pinned simulator source, assets, one Python environment and the frozen checkpoint.
# Usage: bash setup.sh SCRATCH      (Linux x86-64, uv, git, git-lfs, curl, Mesa EGL; about 6 GB)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(realpath -m "${1:?scratch directory}")
mkdir -p "$scratch"/{policy,vlm,libero-config}
cd "$scratch"
if [[ ! -d pro2/.git ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0 pro2
    GIT_LFS_SKIP_SMUDGE=1 git -C pro2 checkout -q e5500c690c1498eca1b7050350654964190cdf70
fi
python3 "$here/hydrate.py" pro2   # SHA-checked LFS payloads from media.githubusercontent.com
sim="$scratch/pro2/simulator-source/libero/libero"
printf '%s: "%s"\n' benchmark_root "$sim" bddl_files "$sim/bddl_files" init_states "$sim/init_files" \
    datasets "$sim/.." assets "$sim/assets" > libero-config/config.yaml
[[ -x env/bin/python ]] || uv venv -q --python 3.12 env
uv pip install -q --python env/bin/python --no-deps --index https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match -r "$here/requirements.txt"
base=https://huggingface.co/lerobot/smolvla_libero/resolve/a24f1f380461b665158d52acd140152c79e8b36d
for file in config.json model.safetensors policy_{pre,post}processor.json \
            policy_preprocessor_step_5_normalizer_processor.safetensors \
            policy_postprocessor_step_0_unnormalizer_processor.safetensors; do
    [[ -s policy/$file ]] || curl -fsSL --retry 2 "$base/$file" -o "policy/$file"
done
base=https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/4cde9d208e19519c36917aa170fffc7e381aa0ad
for file in config.json added_tokens.json chat_template.json generation_config.json merges.txt \
            preprocessor_config.json processor_config.json special_tokens_map.json tokenizer.json \
            tokenizer_config.json vocab.json; do
    [[ -s vlm/$file ]] || curl -fsSL --retry 2 "$base/$file" -o "vlm/$file"
done
sha256sum -c "$here/../frozen-vla-factory-20261123-r1/view/downloads.sha256"
echo "ready: $scratch"
