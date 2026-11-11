# Restore the Baige archives

## Sidney training rollouts

The 16 original `.pt` objects are in private Baige BOS under
`/mnt/bos/adgfyl/strong-plant-archives/storage-audit-baige-20261111-r1/sidney-training-rollouts/`.
The [release receipt](RELEASE.json) maps every original path to its archive
object, SHA256, byte count, owner, mode and modification time. The same receipt
is available as `RELEASE.json` in that BOS directory. Every source directory
has its original rollout JSONs and a `.pt.cold.json` pointer per archived
payload.

Each payload needs approximately 5.4 GiB on PFS. Both arms need 86.2 GiB.
Set the pointer for the desired arm and filename.

```bash
arm=ppo02
payload=iteration-0001-policy-0000-rank-00.pt
base=/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1
pointer="$base/$arm/training-rollouts/$payload.cold.json"
```

The final checkpoints, initial states, sampling records and per-payload JSONs
are already on PFS.

## MIKASA failed qualification archive

The 25.8 GiB tarball is in the sibling `mikasa-failed-qualification/` BOS
directory. Its [release receipt](MIKASA-RELEASE.json) and the original
`qualification-step2-archive.json` list the compressed-object digest and
the original qualification member hashes, respectively. Set the pointer
to restore the tarball at its original path.

```bash
base=/mowenjie/fluffy-bassoon/outputs/mikasa-fullft-reset-s1000-new-20261107-r3
pointer="$base/qualification-step2-preserved.tar.zst.cold.json"
```

To recover the checkpoint inside the restored tarball, extract with
`tar -I zstd -xf "$base/qualification-step2-preserved.tar.zst" -C "$scratch"`
into a fresh empty directory. Compare the extracted members with
`qualification-step2-archive.json` before using them.

## Verified copy back to PFS

Run in a Baige Linux container with PFS and BOS mounted and sufficient free
space. Set `pointer` as above and run the copy for each file you need.

```bash
python3 - "$pointer" <<'PY'
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

Restore the required files before starting a reader.
