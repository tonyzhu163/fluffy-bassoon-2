#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pin=$1
prefix=$2

bash "$here/run.sh" "$pin" collect "${prefix}col"
bash "$here/run.sh" "$pin" train "${prefix}fit" "${prefix}col"
