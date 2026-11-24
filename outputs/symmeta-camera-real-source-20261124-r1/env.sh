# Shared primary reader environment. Source from cache.sh / gpu.sh only.
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
sim=$root/setups/libero-meta-learning-20261117-r1
release=$root/sources/libero-pro2-e5500c69
py=$legacy/envs/openpi-215abfb/bin/python
run=$root/outputs/$(basename "$here")
primary=$root/outputs/symmeta-camera-frame-reader-20261124-r1
reader=symmeta-camera-frame-reader-20261124-r1
# Import exactly the immutable primary reader implementation for shared data, features,
# model, optimizer and uncertainty reduction even when unrelated work advances HEAD.
primarycode=$root/setups/symmeta-perceiver-followups-20261123-r1/src-68cedbfe5468
[[ $(git -C "$primarycode" rev-parse HEAD) == 68cedbfe546836ac6fbb8ec6129192d2deb58f94 ]]
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$primarycode/src:$release:$sim/simulator-source"
for mod in libero-crossarm-20261120-r1 libero-xemb-ur5e-sysid-20261120-r1 libero-xemb-ur5e-scale-20261120-r1 xemb-icml-campaign-20261121-r1 symmeta-wider-family-20261123-r1 symmeta-camera-command-20261123-r1 "$reader"; do
    PYTHONPATH="$PYTHONPATH:$primarycode/outputs/$mod"
done
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config" CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
