# Common native runtime from the CPU-qualified progress experiment.
cue_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$cue_dir/../frozen-transfer-progress-20261124-r1/runtime.sh"
here=$cue_dir
shared="$root/outputs/$(basename "$here")"
