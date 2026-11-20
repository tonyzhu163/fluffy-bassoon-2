#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pin=$1
attempt=$2
[[ $(hostname) == job-* && "$attempt" =~ ^[a-z0-9]+$ ]]

bash "$here/run.sh" "$pin" gate1-smoke "${attempt}smoke"
bash "$here/run.sh" "$pin" gate1 "${attempt}gate1"
