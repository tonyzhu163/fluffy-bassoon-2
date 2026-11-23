# Source from the two launchers; both use the exact native job's immutable checkout.
root=/mowenjie/fluffy-bassoon-2
base_pin=a8a997e7d06bb131d3f9710204c58e106f0f0904
base_name=frozen-vla-native-transfer-20261123-r1
code="$root/outputs/$base_name/src-$base_pin"
native="$code/outputs/$base_name"
[[ $(git -C "$code" rev-parse HEAD) == "$base_pin" ]]
[[ -z $(GIT_OPTIONAL_LOCKS=0 git -C "$code" status --porcelain --untracked-files=no) ]]
source "$native/env.sh"
export PYTHONPATH="$native:$PYTHONPATH"
export NATIVE_HISTORICAL="$native/historical.json"
export NATIVE_MOTION_DIR="$root/outputs/frozen-vla-motion-models-20261123-r1/cpu01"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
test -d /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1/params
# Keep writable process caches, temporary IPC files and receipts in this attempt.
mkdir "$out/cache" "$out/tmp"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache"
export MESA_SHADER_CACHE_DIR="$out/cache/mesa" NUMBA_CACHE_DIR="$out/cache/numba"
