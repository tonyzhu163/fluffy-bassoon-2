#!/usr/bin/env bash
# Assets stay outside Git. All Hub references are immutable revisions.
set -euo pipefail
root=${1:?pass a cache directory}
mkdir -p "$root"
cd "$root"

fetch() {
    local source=$1 dest=$2
    mkdir -p "$(dirname "$dest")"
    if [[ ! -f "$dest" ]]; then
        curl --fail --silent --show-error --location --retry 3 "$source" -o "$dest.part"
        mv "$dest.part" "$dest"
    fi
}

model=https://huggingface.co/lerobot/smolvla_libero/resolve/a24f1f380461b665158d52acd140152c79e8b36d
for f in config.json model.safetensors policy_{pre,post}processor.json \
    policy_preprocessor_step_5_normalizer_processor.safetensors \
    policy_postprocessor_step_0_unnormalizer_processor.safetensors; do
    fetch "$model/$f" "ckpt/$f"
done

vlm=https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/4cde9d208e19519c36917aa170fffc7e381aa0ad
for f in tokenizer.json config.json preprocessor_config.json processor_config.json \
    special_tokens_map.json tokenizer_config.json chat_template.json added_tokens.json merges.txt vocab.json; do
    fetch "$vlm/$f" "vlm/$f"
done

for arm in ur5e iiwa; do
    rev=473f94e10c19660ccf89bf434e283c782c9c8c32
    if [[ $arm == iiwa ]]; then rev=8aaff5f95753f00d15b51d8046efbffcc6446189; fi
    data="https://huggingface.co/datasets/OliverHausdoerfer/libero_object_${arm}_defaultCams/resolve/$rev"
    for f in data/chunk-000/file-000.parquet meta/info.json meta/tasks.parquet meta/episodes/chunk-000/file-000.parquet \
        videos/observation.images.{image,wrist_image}/chunk-000/file-{000,001}.mp4; do
        fetch "$data/$f" "$arm/$f"
    done
done
