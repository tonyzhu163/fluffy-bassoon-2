#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pin=$1
prefix=$2

bash "$here/run.sh" "$pin" gate1-smoke "${prefix}smoke"
bash "$here/run.sh" "$pin" gate1 "${prefix}gate1"
