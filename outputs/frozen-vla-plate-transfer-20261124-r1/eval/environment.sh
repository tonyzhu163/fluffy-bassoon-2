# Source from a detached evaluation checkout on Baige.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$here/../../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
code="$root/outputs/frozen-vla-native-transfer-20261123-r1/src-a8a997e7d06bb131d3f9710204c58e106f0f0904"
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
shared="$root/outputs/frozen-vla-plate-transfer-20261124-r1/eval"
[[ $(git -C "$repo" rev-parse HEAD) == "$EVAL_PIN" && -z $(git -C "$repo" status --porcelain --untracked-files=no) ]]
[[ $(git -C "$code" rev-parse HEAD) == a8a997e7d06bb131d3f9710204c58e106f0f0904 ]]
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export PYTHONPATH="$repo/outputs/libero-paligemma-dualarm-20261120-r2:$repo/outputs/libero-crossarm-20261120-r1:$repo/outputs/frozen-vla-cotraining-20261124-r1/data:$PYTHONPATH"
export PYTHONHASHSEED=0
