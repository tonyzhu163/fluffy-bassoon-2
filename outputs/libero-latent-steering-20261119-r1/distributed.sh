#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out="/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")"
pin=$1
attempt=$2
[[ $(hostname) == job-* && "$attempt" =~ ^[a-z0-9]+$ ]]
test -s "$out/data/index.json"

bash "$here/run.sh" "$pin" fit-adapter "${attempt}fit"
ckpt="$out/${attempt}fit/training/checkpoint.msgpack"
bash "$here/run.sh" "$pin" eval-adapter "${attempt}dev" "$ckpt"
bash "$here/run.sh" "$pin" collect-history "${attempt}hist" "$ckpt"
archive="$out/${attempt}hist/archive/archive.json"
bash "$here/run.sh" "$pin" fit-history "${attempt}enc" "$archive" "$ckpt"
hckpt="$out/${attempt}enc/history-fit/seed-41.msgpack"
bash "$here/run.sh" "$pin" eval-history "${attempt}eval" "$ckpt" "$hckpt" "$archive"
