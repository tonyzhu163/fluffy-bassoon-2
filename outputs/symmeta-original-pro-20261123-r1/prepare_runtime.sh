#!/usr/bin/env bash
# Source this file and call original_pro_paths to select the read-only runtimes.
original_pro_paths() {
    project=/mowenjie/fluffy-bassoon-2
    local legacy=/mowenjie/fluffy-bassoon
    local campaign=xemb-icml-campaign-20261121-r1
    local setup="$project/setups/$campaign"
    runtime="$project/setups/symmeta-original-pro-20261123-r1"
    export ORIGINAL_PRO_ROOT="$runtime/release" ORIGINAL_PRO_ASSETS="$runtime/assets"
    export SIM_PYTHON="$project/setups/libero-meta-learning-20261117-r1/env/bin/python"
    export POLICY_PYTHON="$legacy/envs/openpi-215abfb/bin/python"
    export PI05_CHECKPOINT="$legacy/inputs/pi05-libero-openpi-20261027-r1"
    export SYMMETA_CHECKPOINT="$project/outputs/$campaign/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack"
    export SYMMETA_SOURCE="$setup/source-ff2ddf0a" OPENPI_SOURCE="$legacy/third_party/openpi-215abfb"
    original_tokenizer="$legacy/cache/openpi-215abfb/big_vision/paligemma_tokenizer.model"
    export PYTHONPATH="$ORIGINAL_PRO_ROOT:$setup/cv2-headless-overlay:$SYMMETA_SOURCE/src:$SYMMETA_SOURCE/outputs/$campaign"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export PYTHONUNBUFFERED=1
}

original_pro_cpu() {
    export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
    export MUJOCO_EGL_DEVICE_ID=0 GALLIUM_DRIVER=llvmpipe
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
}

original_pro_caches() {
    export XDG_CACHE_HOME="$1/cache" TMPDIR="$1/tmp"
    export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
    export OPENPI_DATA_HOME="$XDG_CACHE_HOME/openpi"
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$OPENPI_DATA_HOME/big_vision"
    local copy="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
    cp "$original_tokenizer" "$copy"
    printf '%s  %s\n' 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 "$copy" | sha256sum -c -
}

verify_checkout() {
    local path=$1 expected=$2 actual status
    actual=$(git -C "$path" rev-parse HEAD) || return 1
    status=$(git --no-optional-locks -C "$path" status --porcelain) || return 1
    if [[ $actual != "$expected" || -n $status ]]; then
        echo "Unexpected revision or modified files in $path (HEAD $actual)" >&2
        return 1
    fi
}

original_pro_verify() {
    verify_checkout "$ORIGINAL_PRO_ROOT" e235a970758ecd78ecbb585fca6b915a8d44d18d || return 1
    verify_checkout "$ORIGINAL_PRO_ASSETS" 3a33dc4cfbea7ff5eac6764e9c557500972b6259 || return 1
    verify_checkout "$OPENPI_SOURCE" 215abfb217dbac7d5f1273282331b9b1866c0479 || return 1
    verify_checkout "$SYMMETA_SOURCE" ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c || return 1
    [[ -x $SIM_PYTHON && -x $POLICY_PYTHON && -r $original_tokenizer ]]
}

install_checkout() {
    local name=$1 upstream=$2 pin=$3 source tmp
    if [[ -e $runtime/$name ]]; then
        verify_checkout "$runtime/$name" "$pin"
        return
    fi
    source=$upstream
    if [[ -f $runtime/$name.bundle ]]; then source="$runtime/$name.bundle"; fi
    tmp=$(mktemp -d "$runtime/.$name.XXXXXX")
    git clone --no-checkout "$source" "$tmp"
    git -C "$tmp" checkout --detach "$pin"
    verify_checkout "$tmp" "$pin"
    mv -T "$tmp" "$runtime/$name"
}

if [[ ${BASH_SOURCE[0]:-} == "$0" ]]; then
    set -euo pipefail
    [[ $(uname -s) == Linux ]]
    original_pro_paths
    here=$(cd "$(dirname "$0")" && pwd -P)
    case ${1:-verify} in
        install)
            mkdir -p "$runtime"
            exec 9>"$runtime/.prepare.lock"
            flock -x 9
            df -h "$runtime"
            df -i "$runtime"
            probe=$(mktemp "$runtime/.write.XXXXXX")
            echo writable > "$probe"
            [[ $(cat "$probe") == writable ]]
            rm -- "$probe"
            install_checkout release https://github.com/hongyuxiyohung/LIBERO-PRO.git e235a970758ecd78ecbb585fca6b915a8d44d18d
            install_checkout assets https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro 3a33dc4cfbea7ff5eac6764e9c557500972b6259
            original_pro_verify
            ;;
        verify) original_pro_verify ;;
        smoke)
            original_pro_verify
            original_pro_cpu
            receipt=$(mktemp -d "$runtime/cpu-smoke.XXXXXX")
            original_pro_caches "$receipt"
            echo "CPU receipts in $receipt"
            timeout 600 "$POLICY_PYTHON" -Bu "$here/smoke_runtime.py" policy --output "$receipt/policy.json"
            timeout 300 "$SIM_PYTHON" -Bu "$here/smoke_runtime.py" sim --output "$receipt/sim.json"
            ;;
        *) echo "Usage: bash prepare_runtime.sh {install|verify|smoke}" >&2; exit 2 ;;
    esac
fi
