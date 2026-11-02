"""Run-level numerical check: interrupted plateau training equals uninterrupted."""
import json
from pathlib import Path
import socket
import subprocess
import sys

import torch

root = Path('/mowenjie/fluffy-bassoon-2/outputs/opendrawer-ppo-ad-dpt-20261102-r1')
code = Path(__file__).resolve().parents[2]
pin = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
assembly = json.loads((root/'data/assembly.json').read_text())
train, val = assembly['datasets']['ad_train'], assembly['datasets']['ad_validation']
assert not torch.cuda.is_available()
common = [sys.executable,'-B','-m','fluffy_bassoon.meta_rl.vla_history_train',
    '--train-data',train['path'],'--train-sha256',train['sha256'],
    '--validation-data',val['path'],'--validation-sha256',val['sha256'],
    '--mode','full','--width','8','--layers','1','--heads','2','--context-length','4',
    '--max-window','4','--batch-size','2','--eval-every','1','--validation-batches','1',
    '--seed','8','--device','cpu','--threads','2','--source-commit',pin,
    '--expected-hostname',socket.gethostname(),'--until-converged',
    '--minimum-updates','1','--validation-patience','1','--validation-relative-delta','1000000000']
out = root/'cpu-resume-q3'
out.mkdir(exist_ok=False)
subprocess.run(common+['--output-dir',str(out/'direct'),'--max-elapsed-seconds','120'],check=True)
interrupted = common+['--output-dir',str(out/'resumed'),'--max-elapsed-seconds','0.000000001']
subprocess.run(interrupted,check=True)
status = json.loads((out/'resumed/training-status.json').read_text())
assert not status['converged'] and status['update']==1
subprocess.run(interrupted+['--resume',str(out/'resumed/latest.pt')],check=True)
direct = torch.load(out/'direct/latest.pt',map_location='cpu',weights_only=False)
resumed = torch.load(out/'resumed/latest.pt',map_location='cpu',weights_only=False)
assert direct['update']==resumed['update']==2 and direct['plateau']==resumed['plateau']
assert direct['plateau']['converged']
for name in direct['policy']:
    torch.testing.assert_close(direct['policy'][name],resumed['policy'][name],rtol=0,atol=0)
assert direct['rng']['sampler']==resumed['rng']['sampler']
torch.testing.assert_close(direct['rng']['torch'],resumed['rng']['torch'],rtol=0,atol=0)
(out/'result.json').write_text(json.dumps(dict(completed=True,source_commit=pin,
    real_derived_data=True,scientific_fit=False,cuda_hidden=True,updates=2,
    resumed_weights_bitwise_equal=True,sampler_and_torch_rng_equal=True,
    plateau_restored=True),indent=2)+'\n')
