"""CPU-only shell check of the exact multi-rank child supervisor in the deployed launcher."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

src=Path(sys.argv[1]).resolve()
receipt=Path(sys.argv[2])
pin=subprocess.check_output(["git","-C",src,"rev-parse","HEAD"],text=True).strip()
shell=(src/"launch.sh").read_text()
snippet=shell.split('transport/with-cache.sh" bash -c ' + chr(39),1)[1].split(chr(39),1)[0]
with tempfile.TemporaryDirectory(dir=receipt.parent) as d:
    work=Path(d)
    mock=work/"fake-policy"
    mock.write_text('#!/bin/bash\necho "lane $5"\nif [[ $5 == "$FAILED_RANK" ]]; then exit 43; fi\n')
    mock.chmod(0o700)
    count=0
    for bad in ("none","0","1","3"):
        for _ in range(25):
            env=dict(os.environ,n="4",here=str(src),attempt=d,phase="full",py=str(mock),FAILED_RANK=bad,
                     CUDA_VISIBLE_DEVICES="")
            run=subprocess.run(["bash","-c",snippet],env=env,check=False)
            assert run.returncode==(0 if bad=="none" else 43)
            assert all((work/f"rank{i}.log").read_text().strip()==f"lane {i}" for i in range(4))
            count+=1
    receipt.write_text(json.dumps(dict(code=pin,launcher_sha256=hashlib.sha256(shell.encode()).hexdigest(),
        gpu_hours=0,fake_four_rank_trials=count,all_passed=True,
        tested="Simultaneous completions, all-zero exits and individual failing ranks 0, 1, 3"),indent=2)+"\n")
