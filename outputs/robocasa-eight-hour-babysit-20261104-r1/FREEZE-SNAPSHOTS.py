"""Freeze the current best candidates for deployment qualification, without SGD."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    assert sys.platform=='linux'
    import torch
    root=Path('/mowenjie/fluffy-bassoon-2')
    run=root/'outputs/robocasa-eight-hour-babysit-20261104-r1'
    joint=root/'outputs/robocasa-three-task-meta-baseline-20261104-r2/joint'
    output=run/'inputs/direct-snapshot-01';output.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assembly=joint/'preparation/ASSEMBLY.json'
    assert sha(assembly)=='fc3f388ef3feddde4c44ebd45ae9d7e73dab4a0fc58252c506315697be2a4758'
    corpus=json.loads(assembly.read_text());rows={}
    for method in ('ad','dpt','dit','ic_iql'):
        source=joint/'fits'/method/'candidate.pt'
        # Each copy is a new identity; a concurrent atomic candidate replacement
        # is diagnosed and retained rather than publishing a mixed snapshot.
        for attempt in range(3):
            before=sha(source);temporary=output/f'{method}-copy-{attempt:02d}.pt'
            shutil.copyfile(source,temporary);digest=sha(temporary);after=sha(source)
            if before==digest==after:
                target=output/(method+'.pt');temporary.rename(target);break
        else:
            raise RuntimeError('Candidate changed throughout snapshot freeze; preserve copies and diagnose')
        saved=torch.load(target,map_location='cpu',weights_only=False)
        contract=saved['manifest']['contract'];config=saved['config']
        assert contract['source_commit']=='d71cb532f6cf4933a4a5e236e1b085bce74ce969'
        assert contract['objective']==method and contract['mode']=='full'
        assert contract['train_sha256']==corpus['datasets'][method+'_train']['sha256']
        assert contract['validation_sha256']==corpus['datasets'][method+'_validation']['sha256']
        assert tuple(config[k] for k in ('width','layers','heads','context_length'))==(256,6,8,192)
        status=json.loads((joint/'fits'/method/'training-status.json').read_text())
        row=dict(method=method,qualification_only=True,training_continues=True,
            source_assembly_sha256=sha(assembly),candidate=dict(path=str(target),sha256=digest),
            original_candidate=str(source),candidate_manifest=saved['manifest'],
            training_status_at_freeze=status,frozen_shanghai=datetime.datetime.now(datetime.timezone(
                datetime.timedelta(hours=8))).isoformat(),optimizer_updates=0,
            purpose='Two-case ID native deployment/history qualification; no final-model or checkpoint-selection claim')
        (output/(method+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows[method]=row
    (output/'MANIFEST.json').write_text(json.dumps(dict(completed=True,methods=rows,GPU_hours=0,SGD=0),indent=2)+'\n')
    print(json.dumps({key:dict(sha256=value['candidate']['sha256'],update=value['candidate_manifest']['update']) for key,value in rows.items()}))


if __name__=='__main__':
    main()
