set -Eeuo pipefail
export GIT_TERMINAL_PROMPT=0
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WT=$TASK_ROOT/worktrees/libero90-pi05-headroom-screen-20261101-s0-r1
TASK_CONTROL=$TASK_ROOT/outputs/libero90-pi05-headroom-screen-20261101-r1
TASK_SOURCE=4ec35eb9f563813e43293f797e8141a12fb0c4be
TASK_GRANT_PIN=0e487adf2783c9c7bdc01fadd018b297f7f157b2
printf 'DEPLOY_START %s\n' "$(date -u +%FT%TZ)"
mkdir -p "$TASK_CONTROL/launch"
(
flock -w 20 9
git -C "$TASK_ROOT/repo" bundle verify /mowenjie/fluffy-bassoon-2/fb2-headroom-20b0e42-20261101.bundle
git -C "$TASK_ROOT/repo" fetch /mowenjie/fluffy-bassoon-2/fb2-headroom-20b0e42-20261101.bundle HEAD
if [ ! -e "$TASK_WT" ]; then git -C "$TASK_ROOT/repo" worktree add --detach "$TASK_WT" "$TASK_SOURCE"; fi
test "$(git -C "$TASK_WT" rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git -C "$TASK_WT" status --porcelain)"
) 9> "$TASK_ROOT/.headroom-deploy.lock"
python3 - <<'PYVERIFY'
from pathlib import Path
import json,hashlib,shutil,subprocess
root=Path('/mowenjie/fluffy-bassoon-2'); control=root/'outputs/libero90-pi05-headroom-screen-20261101-r1'
wt=root/'worktrees/libero90-pi05-headroom-screen-20261101-s0-r1'
u=shutil.disk_usage(root); assert u.free >= 22*1024**3, u
probe=control/'launch/.write-probe'; probe.write_bytes(b'headroom-screen'); probe.unlink()
raw=subprocess.check_output(['git','-C',str(root/'repo'),'show','0e487adf2783c9c7bdc01fadd018b297f7f157b2:outputs/libero90-pi05-headroom-screen-20261101-r1/GRANT.json'])
assert hashlib.sha256(raw).hexdigest()=='893c368f8285bf18eed28d414bacfa2a45ea1ee6d4b459db49ff744823e987d3'
p=control/'GRANT.json'
if p.exists(): assert p.read_bytes()==raw, 'Existing grant differs'
else: p.write_bytes(raw)
assert not (root/'outputs/libero90-pi05-headroom-screen-20261101-s0-r1').exists(), 'Run already exists; do not duplicate'
paths=['/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python','/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python','/mowenjie/envs/wam-libero-plus-py310/bin/python','/mowenjie/fluffy-bassoon-2/envs/dsrl-pi0-overlay-20261029-r1','/mowenjie/fluffy-bassoon-2/deps/libero-dsrl-87edbd1-20261029-r1/repo']
for name in paths: assert Path(name).exists(), name
receipt=Path('/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1/READY.json')
assert hashlib.sha256(receipt.read_bytes()).hexdigest()=='ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
result=dict(source_commit='4ec35eb9f563813e43293f797e8141a12fb0c4be',grant_sha256=hashlib.sha256(raw).hexdigest(),worktree=str(wt),clean=True,storage_free=u.free,write_probe=True,runtime_paths_present=True,motor_receipt_verified=True,models_loaded=False,simulator_actions=0)
(control/'launch/deployment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PYVERIFY
printf 'DEPLOY_COMPLETE %s\n' "$(date -u +%FT%TZ)"
