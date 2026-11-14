"""Extract only a fixed corrected step30 teacher, checking archive member hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert sys.platform=='linux'
    code=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==a.pin
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    assert a.task in ('OpenDrawer','CoffeeSetupMug','PickPlaceCounterToStove','CloseToasterOvenDoor','TurnOffStove')
    assert sha(a.receipt)==a.receipt_sha256
    receipt=json.loads(a.receipt.read_text())
    assert receipt['verified'] and receipt['all_stageback_members_verified']
    archive=Path(receipt['cold']);assert archive.stat().st_size==receipt['archive_bytes']
    prefix=f'{a.task}/checkpoints/global_step_30/'
    selected={prefix+'pilot-boundary.json':'pilot-boundary.json',
              prefix+'actor/model_state_dict/full_weights.pt':'full_weights.pt'}
    references={r['path']:r for r in receipt['files']}
    assert set(selected)<=references.keys()
    a.output.mkdir(parents=True,exist_ok=False)
    began=time.time();records={}
    with tarfile.open(archive,'r:') as bank:
        for member in bank:
            if member.name not in selected:
                continue
            assert member.isfile() and member.size==references[member.name]['bytes']
            path=a.output/selected[member.name]
            with bank.extractfile(member) as source,path.open('xb') as dest:
                shutil.copyfileobj(source,dest,16*2**20)
            digest=sha(path)
            assert digest==references[member.name]['sha256']
            records[path.name]=dict(path=str(path),sha256=digest,bytes=path.stat().st_size,
                                   source_member=member.name)
            if len(records)==len(selected):break
    assert set(records)==set(selected.values())
    boundary=json.loads((a.output/'pilot-boundary.json').read_text())
    assert boundary['completed'] and boundary['global_step']==30
    assert any(r['sha256']==records['full_weights.pt']['sha256'] for r in boundary['actor_checkpoint_files'])
    result=dict(completed=True,CPU_only=True,task=a.task,teacher_step=30,source_commit=a.pin,
        receipt=str(a.receipt),receipt_sha256=a.receipt_sha256,archive_sha256=receipt['archive_sha256'],
        files=records,all_selected_members_rehashed=True,elapsed_seconds=time.time()-began)
    (a.output/'TEACHER.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
