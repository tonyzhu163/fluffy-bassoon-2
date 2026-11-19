#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out="/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")"
pin=$1
attempt=$2
[[ $(hostname) == job-* && "$attempt" =~ ^[a-z0-9]+$ ]]
test -s "$out/data/full-check.json"

bash "$here/run.sh" "$pin" smoke "${attempt}check"
checkpoint="$out/${attempt}check/training/checkpoint.msgpack"
bash "$here/run.sh" "$pin" smoke-eval "${attempt}live" "$checkpoint"
bash "$here/run.sh" "$pin" fit "${attempt}fit"
checkpoint="$out/${attempt}fit/training/checkpoint.msgpack"
bash "$here/run.sh" "$pin" eval "${attempt}dev" "$checkpoint"
