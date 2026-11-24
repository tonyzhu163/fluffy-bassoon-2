# Common Baige runtime and immutable native-query dependencies.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$here/../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
shared="$root/outputs/$(basename "$here")"
native_pin=a8a997e7d06bb131d3f9710204c58e106f0f0904
native_name=frozen-vla-native-transfer-20261123-r1
code="$root/outputs/$native_name/src-$native_pin"

pinned_tree() {
    [[ $(git -C "$1" rev-parse HEAD) == "$2" && -z $(git --no-optional-locks -C "$1" status --porcelain --untracked-files=no) ]]
}

runtime() {
    [[ $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ ]]
    pinned_tree "$repo" "$pin"
    pinned_tree "$code" "$native_pin"
    source "$code/outputs/$native_name/env.sh"
    pinned_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
    pinned_tree /mowenjie/fluffy-bassoon/third_party/openpi-215abfb 215abfb217dbac7d5f1273282331b9b1866c0479
    export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba"
    export MPLCONFIGDIR="$out/cache/matplotlib" MESA_SHADER_CACHE_DIR="$out/cache/mesa"
    mkdir -p "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
}
