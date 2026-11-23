#!/usr/bin/env bash
# Wrap the already reserved _payload or _lanes command. See ../TRANSPORT.md.
set -euo pipefail
(( $# )) || { echo 'Usage: with-cache.sh COMMAND [ARG ...]' >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
case $(stat -f -c %T /tmp) in
    overlayfs|ext2/ext3|xfs|tmpfs) ;;
    *) echo 'Unqualified /tmp filesystem. Run the canonical payload.' >&2; exit 1 ;;
esac
cache=$(mktemp -d /tmp/reground-transformers.XXXXXXXX)
trap 'rm -rf -- "$cache"' EXIT
/usr/bin/python3 -IB "$here/wheel.py" --stage "$cache" >&2
export REGROUND_TRANSFORMERS_PATH="$cache"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$cache"
"$@"
