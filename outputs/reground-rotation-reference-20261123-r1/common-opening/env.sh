# Source in a fresh process. The scientific code and weights are the original streaming release.
run=outputs/reground-rotation-reference-20261123-r1
project=/mowenjie/fluffy-bassoon-2
export REGROUND_RUNTIME_ROOT=$project/$run REGROUND_UPSTREAM=$project/$run/upstream
root=$REGROUND_RUNTIME_ROOT
science_pin=cab855a03c8ff1a3f1926513369099bf11c4b859
science=$root/src-$science_pin
archive=$project/setups/xemb-icml-campaign-20261121-r1/source-ff2ddf0a
[[ $(git -C "$science" rev-parse HEAD) == "$science_pin" ]]
git -C "$science" diff --quiet HEAD -- "$run" src scripts outputs/symmeta-original-pro-20261123-r1/eval_reader.py
[[ $(git -C "$archive" rev-parse HEAD) == ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c ]]
git -C "$archive" diff --quiet HEAD -- src outputs
source "$root/runtime/activate.sh"
export SYMMETA_SOURCE=$archive SYMMETA_CV2=$project/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay
export SYMMETA_CHECKPOINT=$project/outputs/xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack
export SYMMETA_PYTHON=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
export PYTHONPATH="$science/$run:$PYTHONPATH:$archive/src:$science/src"
export REGROUND_NEXEC=10 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export NUMBA_CACHE_DIR="$attempt/numba-cache" PYTHONPYCACHEPREFIX="$attempt/pycache"
