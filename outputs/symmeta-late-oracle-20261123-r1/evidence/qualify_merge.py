"""Exercise the strict merger with an explicitly synthetic Linux CPU fixture."""
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import records
from paired_records import encode, parse, require
from panel import ARMS, IDENTITY

source = Path(sys.argv[1])
data = records.panel(source)
require(os.environ["CUDA_VISIBLE_DEVICES"] == "", "CPU fixture only")
sha = (source/"panel.sha256").read_text().strip()
mark = "0"*64
# Native shape/dtype-aware digests distinguish the two IEEE signed zeros.
from hashlib import sha256
def array_digest(x):
    return sha256(repr((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
plus_zero = np.zeros((5,7),dtype=np.float32)
minus_zero = plus_zero.copy()
minus_zero[:,6] *= -1.
require(array_digest(plus_zero) != array_digest(minus_zero) and np.array_equal(plus_zero,minus_zero),
        "Signed-zero regression fixture")
rejections = []
native_validate = records.validate


def rejects(out, why):
    try:
        records.summary(out)
    except ValueError:
        rejections.append(why)
    else:
        raise AssertionError(why)


with TemporaryDirectory(prefix="reducer-only-", dir=source) as temp:
    out = Path(temp)
    for name in ("panel.json","panel.sha256"):
        shutil.copyfile(source/name,out/name)
    (out/"cases").mkdir()
    for r in data["rows"]:
        ref = data["conditions"][r["shard"]]
        reset = dict(ref["gpu_reset"], renderer="CPU arithmetic fixture", renderer_vendor="CPU arithmetic fixture")
        arms = {}
        for name in ARMS:
            success = ((r["shard"]%2 == 0 and r["index"]%12 != 0) if name == "exact1" else
                       (name == "stream11" and r["index"]%12 in (4,11)))
            chunks = []
            for j,n in enumerate((5,5,1)):
                exact = name == "exact1" or (name == "exact11" and j == 2)
                c = dict(start=j*5+1,steps=n,correction=r["truth"] if exact else list(IDENTITY),
                         noise=ref["noise_draws"][j],proposal=mark,issued=mark,canonical=mark,changed_offsets=[])
                if exact and r["truth"] == [0.,1.,-1] and j < 2:
                    c.update(canonical=array_digest(plus_zero),issued=array_digest(minus_zero))
                if j<2:
                    c.update(input=mark,after=mark)
                chunks.append(c)
            arms[name] = dict(reset=reset, released_prompt=reset["prompt"], controls=11, success=success,
                truncated=False, terminated=True, native_return=int(success), native_rewards=[0.]*10+[float(success)],
                chunks=chunks, opening_proposal_sha256=mark)
        (out/"cases"/f"{r['index']:02d}.json").write_bytes(encode(dict(row=r,panel_sha256=sha,arms=arms)))
    # The one substitution selects CPU-renderer validation for the fixture. All
    # production layout, outcome, correction, prefix and bank checks still run.
    with patch.object(records,"validate",lambda r,a,f: native_validate(r,a,f,cpu=True)):
        last = out/"cases/95.json"
        raw = last.read_bytes()
        last.unlink()
        rejects(out,"missing case")
        last.write_bytes(raw)
        extra = out/"cases/96.json"
        extra.write_bytes(raw)
        rejects(out,"unexpected case")
        extra.unlink()
        wrong = parse(raw)
        wrong["row"]["index"] = 1
        last.write_bytes(encode(wrong))
        rejects(out,"wrong row")
        last.write_bytes(raw)
        records.summary(out)
    result = parse((out/"summary.json").read_bytes())
    require(result["all"]["native_successes"] == dict(exact1=44,exact11=0,stream11=16)
            and result["ood"]["native_successes"] == dict(exact1=32,exact11=0,stream11=16), "Fixture counts")
    onset, reader, total = result["ood"]["contrasts"]
    require(onset["difference_pp"] == 50 and onset["task_ci95_pp"] == [12.5,87.5]
            and reader["difference_pp"] == -25 and reader["task_ci95_pp"] == [-25,-25]
            and total["difference_pp"] == 25, "Direction, magnitude or cluster interval")
    print(json.dumps(dict(status="passed",fixture_only=True,scored_native_episodes=0,
          tested_triplets=96,signed_zero_numerically_unchanged=True,rejected=rejections,ood_fixture_contrasts=result["ood"]["contrasts"]),sort_keys=True))
