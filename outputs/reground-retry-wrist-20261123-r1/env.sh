# Source with bash, after setting here, attempt and RETRY_PIN.
set -eu
export GIT_OPTIONAL_LOCKS=0
test "$(git -C "$here" rev-parse HEAD)" = "$RETRY_PIN"
git -C "$here" diff --quiet HEAD -- .
project=/mowenjie/fluffy-bassoon-2
name=outputs/reground-rotation-reference-20261123-r1
root=$project/$name
control_pin=c3efd5042a29c68248c07b2d6375d367df6378ef
control=$root/src-$control_pin
test "$(git -C "$control" rev-parse HEAD)" = "$control_pin"
git -C "$control" diff --quiet HEAD -- "$name/common-opening" "$name/transport"
(cd "$root" && sha256sum --quiet -c "$control/$name/speed/runtime.sha256")
# Sets pinned reference, reader archive, CV2/JAX interpreter, native environment and ten-control cadence.
source "$control/$name/common-opening/env.sh"
export PYTHONPATH="$here:$control/$name/common-opening:$PYTHONPATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
libs=$project/setups/libero90-full-action-20261119-r1/desktop-libs02
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu:$libs/prefix/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
py=$root/env/bin/python
