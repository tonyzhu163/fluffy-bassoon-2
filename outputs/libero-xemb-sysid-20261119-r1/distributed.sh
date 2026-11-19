#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out="/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")"
pin=$1
attempt=$2
[[ $(hostname) == job-* && "$attempt" =~ ^[a-z0-9]+$ ]]

bash "$here/run.sh" "$pin" test "${attempt}test"
bash "$here/run.sh" "$pin" smoke "${attempt}smoke"
bash "$here/run.sh" "$pin" gate1 "${attempt}gate1"
bash "$here/run.sh" "$pin" collect "${attempt}col"
archive="$out/${attempt}col/archive/archive.json"
bash "$here/run.sh" "$pin" train "${attempt}trn" "$archive"
train_dir="$out/${attempt}trn/train"
bash "$here/run.sh" "$pin" eval-live "${attempt}eval" "$archive" "$train_dir"
