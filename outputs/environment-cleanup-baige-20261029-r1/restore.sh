#!/usr/bin/env bash
set -euo pipefail
sha256sum -c /mowenjie/fluffy-bassoon-2/outputs/environment-cleanup-baige-20261029-r1/archive.sha256
cd /mowenjie/fluffy-bassoon-2/envs
for e in dsrl-can-20261029-r1 dsrl-can-20261029-r2 metaworld-baselines-20261028-r1 robotwin-baseline-smoke-20261028-r1; do test ! -e "$e"; done
tar --use-compress-program=/root/miniforge/bin/zstd -xf /mowenjie/fluffy-bassoon-2/outputs/environment-cleanup-baige-20261029-r1/retired-envs.tar.zst
