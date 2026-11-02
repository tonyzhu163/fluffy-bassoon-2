"""Run the finite retained-action diagnosis on allocated Linux CPU."""
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import zipfile

assert platform.system() == 'Linux'
code = Path(sys.argv[1]).resolve()
run = 'robocasa-flora-drawer-extra-failures-20261103-r1'
root = Path('/mowenjie/fluffy-bassoon-2')
out = root/'outputs'/run
source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
contract = json.loads((code/'outputs'/run/'INPUTS.json').read_text())
profile_path = code/'outputs/robocasa-opendrawer-rubric-diagnosis-20261102-r1/REPLAY-SOURCE.json'
assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == contract['native_profile_sha256']
env = dict(os.environ, **json.loads(profile_path.read_text())['native_profile'])
env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
space = os.statvfs(root)
assert space.f_bavail*space.f_frsize > 8<<30
start = time.time()
subprocess.run([sys.executable,'-B',str(code/'outputs'/run/'EXTRACT.py')],env=env,check=True,timeout=240)
probe = out/'write-probe'
probe.write_text('owned CPU output write probe\n')
probe.unlink()
subprocess.run([sys.executable,'-B',str(code/'outputs'/run/'REPLAY.py'),'--code',str(code)],env=env,check=True,timeout=600)
summary = json.loads((out/'physics-replay/SUMMARY.json').read_text())
assert summary['episodes'] == 4 and summary['replay_controls'] <= 3000
receipt = dict(completed=True, source_commit=source, hostname=socket.gethostname(),
    elapsed_seconds=time.time()-start, episodes=4, qualified=summary['qualified'],
    retained_replay_controls=summary['replay_controls'], newly_sampled_actions=0,
    motor_calls=0, optimizer_steps=0, gpu_calls=0, cuda_visible_devices='',
    limitations='No saved full MuJoCo state; metadata and all saved16D state boundaries checked.')
(out/'CPU-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
paths = sorted([*out.glob('*.json'), *(out/'physics-replay').glob('*.json')])
manifest = dict(source_commit=source, files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
archive = Path('/root/workspace/drawer-extra-failures-r1-evidence.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.write(p,str(p.relative_to(out)))
    z.writestr('TRANSFER-MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print('ARCHIVE',archive.name,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
print('CPU_RECEIPT',json.dumps(receipt),flush=True)
for row in summary['results']:
    print(json.dumps({k:row.get(k) for k in ('id','qualified','first_success_control','final_success','max_state_error','reason')}),flush=True)
