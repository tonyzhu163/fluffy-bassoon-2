"""Run with NumPy on the Baige archive host. Reads saved tapes only."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path("/mowenjie/fluffy-bassoon-2/outputs")
menu = root / "libero-decision-pretraining-20261119-r1/menu02/episodes/fitting"
identities = {}
openings = {}
tapes = []
for timing in ("BASE", "S20"):
    folder = menu / f"libero_goal-5-r1-{timing}"
    for mode in ("support", "default", "short_wait", "paced", "gripper_wait"):
        record = json.loads((folder / mode / "result.json").read_text())
        path = folder / mode / "public.npz"
        with np.load(path) as data:
            steps = np.flatnonzero(data["rewards"] > 0) + 1
            assert bool(len(steps)) == record["success"]
            assert len(data["issued"]) == record["controls"]
            role = "support" if mode == "support" else "query"
            assert record["identity"] == identities.setdefault(role, record["identity"])
            assert record["opening_sha256"] == openings.setdefault(role, record["opening_sha256"])
        tapes.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          mode=mode, case=record["case"], controls=record["controls"],
                          first_native_hit=int(steps[0]) if len(steps) else None))
print(json.dumps(dict(identities=identities, opening_hashes=openings, tapes=tapes), indent=2))
