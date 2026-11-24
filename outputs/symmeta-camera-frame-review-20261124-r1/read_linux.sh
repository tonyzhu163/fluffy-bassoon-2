#!/usr/bin/env bash
# Feed a read-only reducer to the project environment on the one available CPU.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
stage=${1:?inputs, fit, source, source_orbit, digital or physical}
revision=${2:?full reader pin}
[[ $revision =~ ^[0-9a-f]{40}$ ]]
root=/mowenjie/fluffy-bassoon-2
code=$root/setups/symmeta-perceiver-followups-20261123-r1/src-${revision:0:12}
run=$root/outputs/symmeta-camera-frame-reader-20261124-r1
geometry=$root/outputs/symmeta-camera-pose-pairs-20261124-r1/cpu01
fit=${3:-fit01}
scored=${4:-physical01}
[[ $fit =~ ^[a-zA-Z0-9_-]+$ && $scored =~ ^[a-zA-Z0-9_-]+$ ]]
sim=$root/setups/libero-meta-learning-20261117-r1
overlay=$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay
libs=$overlay:$code/src:$root/sources/libero-pro2-e5500c69:$sim/simulator-source
for project in libero-crossarm-20261120-r1 libero-xemb-ur5e-sysid-20261120-r1 libero-xemb-ur5e-scale-20261120-r1 xemb-icml-campaign-20261121-r1 symmeta-wider-family-20261123-r1 symmeta-camera-command-20261123-r1 symmeta-camera-frame-reader-20261124-r1; do
    libs=$libs:$code/outputs/$project
done
case $stage in
    inputs) file=check_inputs.py; args=("$run/prep01/bank" "$run/offset01/bank");;
    fit) file=check_fit.py; args=("$run/$fit" "$run/prep01/bank" "$run/offset01/bank");;
    source) file=source_factors.py; args=("$run/$fit" "$run/prep01/bank" "$run/offset01/bank");;
    source_orbit) file=source_orbit.py; args=("$run/$fit" "$geometry/source/bank.json");;
    digital) file=check_scores.py; args=("$run/$fit/digital-predictions.json.gz");;
    physical) file=check_scores.py; args=("$run/$scored/physical-predictions.json.gz" --bank "${5:?absolute physical bank path}");;
    native) file=native_outcomes.py; args=("$root/outputs/symmeta-camera-frame-native-20261124-r1/$scored" "$run/phyGpu01/physical-predictions.json.gz");;
    *) exit 2;;
esac
blas=""
if [[ $stage == digital ]]; then blas=OPENBLAS_CORETYPE=Sandybridge; fi
printf -v args_q " %q" "${args[@]}"
ssh -o ConnectTimeout=8 baige-dev "taskset -c 47 env $blas CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LIBERO_PRO_ROOT=$sim/simulator-source LIBERO_CONFIG_PATH=$sim/libero-config CROSSARM_RELEASE=$root/sources/libero-pro2-e5500c69 PYTHONPATH=$libs /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B -$args_q" < "$here/$file"
