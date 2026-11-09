#!/usr/bin/env bash
# Reproduce the isolated model dependency overlay; never modify a shared env.
set -Eeuo pipefail
TASK_ENV=${1:?new environment path}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_SHARED=/mowenjie/fluffy-bassoon/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/lib/python3.11/site-packages
[[ $(uname -s) == Linux && ! -e $TASK_ENV ]]
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/motus-reference-uv-cache}
"$TASK_ROOT/tools/uv-20261028/bin/uv" venv \
 --python "$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python" "$TASK_ENV"
"$TASK_ROOT/tools/uv-20261028/bin/uv" pip install --python "$TASK_ENV/bin/python" \
 --no-deps --index-url https://pypi.org/simple transformers==4.53.2
export TASK_ENV TASK_SHARED
"$TASK_ENV/bin/python" - <<'PY'
import os,shutil
from pathlib import Path
site=Path(os.environ['TASK_ENV'])/'lib/python3.11/site-packages'
shared=Path(os.environ['TASK_SHARED'])
(site/'motus_shared_dependencies.pth').write_text(str(shared)+'\n')
source=shared/'openpi/models_pytorch/transformers_replace'
files=list(source.rglob('*.py'));assert len(files)==5
for path in files:
 destination=site/'transformers'/path.relative_to(source)
 destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
PY
PYTHONDONTWRITEBYTECODE=1 "$TASK_ENV/bin/python" - <<'PY'
import transformers
from transformers.models.siglip import check
assert transformers.__version__=='4.53.2'
assert check.check_whether_transformers_replace_is_installed_correctly()
print('OpenPI Transformers replacement check passed')
PY
