study_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$study_dir/../frozen-transfer-progress-20261124-r1/runtime.sh"
here=$study_dir
shared="$root/outputs/$(basename "$here")"
if git -C "$repo" symbolic-ref -q HEAD >/dev/null; then
    echo "Native study requires a detached checkout" >&2
    exit 1
fi
