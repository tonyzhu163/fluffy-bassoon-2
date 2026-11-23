#!/usr/bin/env bash
# Run in a parent-scheduled, single-GPU Baige Distributed Training container.
set -euo pipefail
pin=7518022c88711815c8207b4397e4ce84036c6134
run=frozen-vla-native-controller-20261123-r1
src=/mowenjie/fluffy-bassoon-2/outputs/$run/src-$pin
exec bash "$src/outputs/$run/run.sh" "$pin" "${1:-gpu01}"
