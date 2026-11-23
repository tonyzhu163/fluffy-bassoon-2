# Shared verified native benchmark and policy implementation; keep outputs and caches in this run.
root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/frozen-vla-supported-correction-20261123-r1"
native_pin=a8a997e7d06bb131d3f9710204c58e106f0f0904
code="$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$native_pin"
native="$code/outputs/frozen-vla-native-transfer-20261123-r1"
[[ $(git -C "$code" rev-parse HEAD) == "$native_pin" && -z $(git -C "$code" diff --name-only HEAD) ]]
source "$native/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
export PYTHONPATH="$native:$PYTHONPATH"
export NUMBA_CACHE_DIR="$out/cache/numba" MESA_SHADER_CACHE_DIR="$out/cache/mesa" XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp"
mkdir -p "$TMPDIR"
matrix="$run/cpu01/adapters.json"
echo "ad36434d3e748583ed3ca338d26b782fe93aa437652e24abf9013be166edc1a5  $matrix" | sha256sum -c -
