"""Pre-mask visibility diagnostics in an owned stdlib worker, not GPU admission."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def observe(*,environment=None,device_nodes=None,executable=None,timeout=3):
    environment=dict(os.environ if environment is None else environment)
    devices=sorted(str(p) for p in Path('/dev').glob('nvidia[0-9]*')) if device_nodes is None else device_nodes
    binary=shutil.which('nvidia-smi') if executable is None else executable
    smi=dict(path=binary,returncode=None,stdout='',stderr='',exec_error=None)
    if binary:
        path=Path(binary);raw=path.read_bytes()
        smi.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),header_hex=raw[:64].hex())
        try:
            result=subprocess.run([binary,'-L'],capture_output=True,text=True,errors='replace',
                                  timeout=timeout,env=environment,check=False)
            smi.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except OSError as exc:
            if exc.errno!=errno.ENOEXEC:raise
            smi['exec_error']=dict(errno=exc.errno,error=type(exc).__name__,message=str(exc))
    return dict(schema='robotwin_pre_mask_visibility_v1',device_nodes=devices,
        environment_before_mask={key:environment.get(key) for key in ('CUDA_VISIBLE_DEVICES',
            'NVIDIA_VISIBLE_DEVICES','SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_GPUS_ON_NODE')},
        nvidia_smi=smi,allocated_GPU_count_inferred=False,
        note='Missing/ENOEXEC utility is a retained diagnostic; actual allocation comes from owner UI proof')


def validate(observation):
    smi=observation['nvidia_smi']
    if (observation['device_nodes'] or any(line.startswith('GPU ')
            for line in (smi['stdout']+'\n'+smi['stderr']).splitlines())):
        raise ValueError('GPU visible before masking')
    if observation['environment_before_mask']['NVIDIA_VISIBLE_DEVICES'] not in (None,'','none','void'):
        raise ValueError('Nonempty GPU namespace before masking')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();observation=observe()
    # Save exact observations even if visibility validation fails. Unexpected
    # exec errors/timeouts propagate; no guardian/cleanup exception is caught.
    temporary=args.output.with_name(args.output.name+'.partial')
    with temporary.open('x') as stream:
        json.dump(observation,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(temporary,args.output);temporary.unlink()
    validate(observation)


if __name__=='__main__':main()
