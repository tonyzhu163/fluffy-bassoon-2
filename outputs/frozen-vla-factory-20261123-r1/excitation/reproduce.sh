#!/usr/bin/env bash
# Recompute all native physics and scores outside the archived evidence directory.
set -euo pipefail
python_bin=$(realpath -s "${1:?usage: reproduce.sh /venv/bin/python /release/checkout /output/dir}")
export CROSSARM_RELEASE
CROSSARM_RELEASE=$(realpath "${2:?provide the pinned LIBERO-Pro2.0 checkout}")
mkdir -p "${3:?provide an output directory}"
out=$(realpath "$3")
here=$(dirname "$(realpath "$0")")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CROSSARM_RELEASE/simulator-source:$CROSSARM_RELEASE"
export LIBERO_CONFIG_PATH="$out/libero-config"
mkdir -p "$LIBERO_CONFIG_PATH"
"$python_bin" - <<'PY'
import os
from pathlib import Path
import yaml

base = Path(os.environ["CROSSARM_RELEASE"]) / "simulator-source/libero/libero"
paths = dict(benchmark_root=base, bddl_files=base / "bddl_files",
             init_states=base / "init_files", assets=base / "assets",
             datasets=Path(os.environ["LIBERO_CONFIG_PATH"]))
(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").write_text(
    yaml.safe_dump({k: str(v) for k, v in paths.items()}))
PY
"$python_bin" "$here/collect.py" --out "$out/supports.json.gz"
"$python_bin" "$here/score.py" --supports "$out/supports.json.gz" --out "$out/metrics.json"
