#!/usr/bin/env bash
# Reproduce the isolated model dependency overlay; never modify a shared env.
set -Eeuo pipefail
TASK_ENV=${1:?new environment path}
TASK_OIDN_LIB=${2:?official OIDN2.3.3 library directory}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_SHARED=/mowenjie/fluffy-bassoon/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/lib/python3.11/site-packages
[[ $(uname -s) == Linux && ! -e $TASK_ENV ]]
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/motus-reference-uv-cache}
"$TASK_ROOT/tools/uv-20261028/bin/uv" venv \
 --python "$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python" "$TASK_ENV"
"$TASK_ROOT/tools/uv-20261028/bin/uv" pip install --python "$TASK_ENV/bin/python" \
 --no-deps --index-url https://pypi.org/simple transformers==4.53.2
export TASK_ENV TASK_SHARED TASK_OIDN_LIB
"$TASK_ENV/bin/python" - <<'PY'
import os,shutil
from pathlib import Path
site=Path(os.environ['TASK_ENV'])/'lib/python3.11/site-packages'
shared=Path(os.environ['TASK_SHARED'])
simulator=Path('/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/lib/python3.11/site-packages')
curobo=Path('/mowenjie/fluffy-bassoon-2/robotwin-curobo-20261108-cpumotuscurobo03/env/lib/python3.11/site-packages')
(site/'motus_shared_dependencies.pth').write_text(str(curobo)+'\n'+str(simulator)+'\n'+str(shared)+'\n')
source=shared/'openpi/models_pytorch/transformers_replace'
files=list(source.rglob('*.py'));assert len(files)==5
for path in files:
 destination=site/'transformers'/path.relative_to(source)
 destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
sapien=shared/'sapien'
shutil.copytree(sapien,site/'sapien',symlinks=True,
 ignore=shutil.ignore_patterns('__pycache__','oidn_library','_oidn_tricks.py'),
 copy_function=lambda src,dst:Path(dst).symlink_to(Path(src).absolute()))
(site/'sapien.libs').symlink_to(shared/'sapien.libs',target_is_directory=True)
loader=(sapien/'_oidn_tricks.py').read_text()
assert loader.count('2.0.1')==2
(site/'sapien/_oidn_tricks.py').write_text(loader.replace('2.0.1','2.3.3'))
(site/'sapien/oidn_library').mkdir()
for name in ['libOpenImageDenoise.so.2.3.3','libOpenImageDenoise_core.so.2.3.3',
             'libOpenImageDenoise_device_cuda.so.2.3.3']:
 source=Path(os.environ['TASK_OIDN_LIB'])/name;assert source.is_file()
 (site/'sapien/oidn_library'/name).symlink_to(source)
PY
PYTHONDONTWRITEBYTECODE=1 "$TASK_ENV/bin/python" - <<'PY'
import transformers
from transformers.models.siglip import check
assert transformers.__version__=='4.53.2'
assert check.check_whether_transformers_replace_is_installed_correctly()
print('OpenPI Transformers replacement check passed')
import sapien
from sapien import _oidn_tricks
assert _oidn_tricks.oidn_dll._name.endswith('libOpenImageDenoise.so.2.3.3')
print('Isolated SAPIEN OIDN2.3.3 shared libraries loaded')
PY
