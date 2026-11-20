#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pin=$1
attempt=$2
col_attempt=$3
fit_attempt=$4

bash "$here/run.sh" "$pin" eval-live "$attempt" "$col_attempt" "$fit_attempt"
