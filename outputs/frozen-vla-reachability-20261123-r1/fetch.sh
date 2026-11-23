#!/usr/bin/env bash
# Fetch original LIBERO demonstrations. Raw image-bearing HDF5s are scratch inputs.
set -euo pipefail
dest=${1:?destination directory}
mkdir -p "$dest"
pin=f13aa24a3da8c43c7225569f28c562979fa0e35a
base="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/$pin"
while read -r sha path; do
    out="$dest/${path##*/}"
    if [[ ! -f "$out" ]]; then
        curl --fail --location --retry 3 --silent --show-error "$base/$path" -o "$out.part"
        mv "$out.part" "$out"
    fi
    echo "$sha  $out" | shasum -a 256 -c -
done <<'FILES'
1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8 libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5
e329bb21a8ded3457854faf6a23513c90cb4b34f0e40f3f4e9e70451fc9ba504 libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5
ee58a7d72c143a71bf1448674e70940ac7781c1d1bb10a1b0c7a5c69b26abbef libero_goal/put_the_cream_cheese_in_the_bowl_demo.hdf5
FILES
