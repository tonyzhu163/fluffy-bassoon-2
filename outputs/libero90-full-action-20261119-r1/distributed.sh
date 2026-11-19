#!/usr/bin/env bash
# Keep the registered collector and both learner recipes on their qualified source pin.
set -euo pipefail
stage=${1:?Choose qualify or panel}
d=libero90-full-action-20261119-r1
root=/mowenjie/fluffy-bassoon-2
out="$root/outputs/$d"
setup="$root/setups/$d"
pin=4750c0971e6ba9b467faa47aac0dd8641f7d5076
runner="$setup/source-${pin:0:8}/outputs/$d/runner.sh"
rank=${RANK:-0}
case "$stage" in qualify|panel) ;; *) exit 2;; esac
[[ $(hostname) == job-* && "$rank" =~ ^[0-3]$ ]]
attempt=$(mktemp -d "$out/container-$(hostname)-XXXXXX")
exec > >(tee -a "$attempt/console.log") 2>&1
date -Is
# The image exports its Python 3.12 Torch libraries ahead of the Python 3.10
# simulator's own extension RUNPATH. Retain the accelerator driver paths.
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
kept=()
for path in "${paths[@]}"; do
    [[ -n "$path" && ! -f "$path/libtorch_python.so" ]] && kept+=("$path")
done
glvnd="$setup/glvnd01/prefix/usr/lib/x86_64-linux-gnu"
test -r "$glvnd/libOpenGL.so.0"
export LD_LIBRARY_PATH="$glvnd$(printf ':%s' "${kept[@]}")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/mowenjie/envs/wam-libero-plus-py310/bin/python -B -c 'import pathlib,torch
loaded = {s.split()[-1] for s in pathlib.Path("/proc/self/maps").read_text().splitlines() if "/libtorch_python.so" in s}
expected = pathlib.Path(torch.__file__).parent / "lib/libtorch_python.so"
assert {p.resolve() for p in map(pathlib.Path, loaded)} == {expected.resolve()}, loaded
print("Simulator Torch:", expected, flush=True)'
if [[ "$stage" == qualify ]]; then
    exec bash "$runner" "$pin" fitting "$out/corpus01" --shard-index 0 --shards 960
fi
bash "$runner" "$pin" fitting "$out/corpus01" --shard-index "$rank" --shards 4
weights="$out/fit01/weights/frozen"
until [[ -e "$weights/config.json" ]]; do
    if [[ -e "$out/fit01/exit.txt" ]]; then
        read -r rc _ < "$out/fit01/exit.txt"
        [[ "$rc" == 0 && -e "$weights/config.json" ]] || exit 1
    fi
    sleep 30
done
test -s "$weights/meta.pt" && test -s "$weights/imitation.pt"
for phase in live-development live-test; do
    bash "$runner" "$pin" "$phase" "$out/panel01" --shard-index "$rank" --shards 4 \
        --meta "$weights/meta.pt" --imitation "$weights/imitation.pt"
done
