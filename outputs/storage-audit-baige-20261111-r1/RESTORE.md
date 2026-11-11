# Restore archived Sidney training rollouts

The 16 original `.pt` objects are in private Baige BOS under
`/mnt/bos/adgfyl/strong-plant-archives/storage-audit-baige-20261111-r1/sidney-training-rollouts/`.
The [release receipt](RELEASE.json) maps every original path to its archive
object, SHA256, byte count, owner, mode and modification time. The same receipt
is available as `RELEASE.json` in that BOS directory. Every source directory
has its original rollout JSONs and a `.pt.cold.json` pointer per archived
payload.

Run the following in a Baige Linux container with PFS and BOS mounted. Give
it an arm and payload filename from the receipt. Check available space first.
Each payload needs approximately 5.4 GiB on PFS. Both arms need 86.2 GiB.

```bash
arm=ppo02
payload=iteration-0001-policy-0000-rank-00.pt
base=/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1
python3 - "$base/$arm/training-rollouts/$payload.cold.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

receipt = json.loads(Path(sys.argv[1]).read_bytes())
source = Path(receipt["source"])
archive = Path(receipt["archive"])
metadata = receipt["source_stat"]
if source.exists() or source.is_symlink():
    raise FileExistsError(source)
with tempfile.NamedTemporaryFile(dir=source.parent, prefix=".restore-") as staged:
    with archive.open("rb") as cold:
        shutil.copyfileobj(cold, staged, length=8 * 1024**2)
    staged.flush()
    os.fsync(staged.fileno())
    staged.seek(0)
    checksum = hashlib.sha256()
    for block in iter(lambda: staged.read(8 * 1024**2), b""):
        checksum.update(block)
    if staged.tell() != receipt["bytes"] or checksum.hexdigest() != receipt["sha256"]:
        raise RuntimeError(f"Readback failed for {archive}")
    os.fchown(staged.fileno(), metadata["st_uid"], metadata["st_gid"])
    os.fchmod(staged.fileno(), stat.S_IMODE(metadata["st_mode"]))
    os.utime(staged.name, ns=(metadata["st_mtime_ns"], metadata["st_mtime_ns"]))
    os.link(staged.name, source)
print(source)
PY
```

Re-run for each required file before starting a reader. The final checkpoints,
initial states, sampling records and per-payload JSONs are already on PFS.
