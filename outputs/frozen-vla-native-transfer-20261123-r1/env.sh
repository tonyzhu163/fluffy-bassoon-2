# Inherit the scheduler's allocated devices. The CPU qualifier supplies Mesa itself.
root=/mowenjie/fluffy-bassoon-2
sim="$root/setups/libero-meta-learning-20261117-r1"
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
campaign="$root/setups/xemb-icml-campaign-20261121-r1"
export CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69" LIBERO_PRO_ROOT="$sim/simulator-source"
export LIBERO_CONFIG_PATH="$sim/libero-config" OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/cache/openpi-215abfb
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
drivers=
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f "$path/libtorch_python.so" ]]; then drivers+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
export PYTHONPATH="$campaign/cv2-headless-overlay:$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1"
python=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
