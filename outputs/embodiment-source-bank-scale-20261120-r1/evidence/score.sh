#!/usr/bin/env bash
set -euo pipefail
root=/mowenjie/fluffy-bassoon-2
n=libero-paligemma-dualarm-20261120-r2
pin=e2f794c41c9141a7aadaf67339da919336362f22
launch="$root/setups/$n/source-e2f794c4/outputs/$n/run.sh"
campaign="$root/outputs/$n"
fit="$campaign/scale3fit01/fit"
panel="$campaign/large-panel-hostscale301"
reference="$campaign/large-panel-hostbroad01/dev/manifest.json"
audit="$root/outputs/embodiment-source-bank-scale-20261120-r1/evidence/fit.json"
for path in "$panel" "$campaign/scale3prep01" "$campaign/scale3score01" "$campaign/scale3sum01"; do
    test ! -e "$path"
done
printf '%s  %s\n' 8a259ccbed50e6f7cbce3800512cf0142648b998179704df955c52f57cf09169 "$launch" \
    6492662090c28905153f55814d54063f0bbbc3e788dad1678c54bd83e1fe2143 "$reference" \
    "$1" "$audit" "$2" "$fit/candidates/step-006400.pt" "$3" "$fit/manifest.json" | sha256sum -c -

python - "$campaign" "$audit" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
campaign, audit, ck, fit_id=sys.argv[1:]
data=json.loads(Path(audit).read_bytes())
assert data['updates']==6400 and data['exit'][0]=='0' and data['eval_overlap_count']==0
assert data['checkpoints']['scale3fit01']['checkpoint_sha256']==ck
assert data['checkpoints']['scale3fit01']['fit_id']==fit_id
assert not list(Path(campaign).glob(f'large-panel*/dev/scores/{ck}/stride-5'))
PY

bash "$launch" "$pin" scale3prep01 sim large_panel.py prepare --phase dev --root "$panel" --fit "$fit"
python - "$panel/dev/manifest.json" "$reference" "$3" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
path, ref=map(Path,sys.argv[1:3])
panel, reference=[json.loads(p.read_bytes()) for p in (path,ref)]
assert panel['preparation_fit_sha256']==sys.argv[3]
assert panel['phase']==reference['phase']=='dev' and panel['required']==38
assert [x['layout_seed'] for x in panel['layouts']]==list(range(740000,740050))
keys=('layout_seed','layout_id','initial_state_sha256','scene_hash','movable_xy_hash','mobile_joint_xy_hash','file_sha256')
assert panel['released']==reference['released'] and panel['rejected']==reference['rejected']
for row, old in zip(panel['layouts'],reference['layouts'],strict=True):
    assert all(row[k]==old[k] for k in keys)
    assert sha(path.parent/row['state_file'])==sha(ref.parent/old['state_file'])==row['file_sha256']
print('Verified the same 50 registered donor files and four geometry fingerprints.',flush=True)
PY
bash "$launch" "$pin" scale3score01 vlm large_panel.py source --phase dev --root "$panel" --fit "$fit" --step 6400 --stride 5 --workers 8 --arm both
bash "$launch" "$pin" scale3sum01 sim large_panel.py aggregate --phase dev --root "$panel" --fit "$fit" --step 6400 --stride 5
