"""Archived-tape statistics: fit data volume and the Kinova3-vs-IIWA opening gap under shared scene and noise.
Run from the repository root. Reads committed command tapes only; no policy, no simulator."""
import json
from pathlib import Path
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
a = tarfile.open(HERE.parent / "frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz")
T = {}
for arm in ("IIWA", "Kinova3"):
    for l in (34, 35, 36, 37):
        t = json.load(a.extractfile(f"gpu01/episodes/{arm}-{l}-raw.json"))
        T[arm, l] = (np.array([r for c in t["calls"] for r in c["sent"]], np.float32), t["native"]["success"])
out = dict(tapes={f"{k[0]}-{k[1]}": dict(controls=len(v[0]), success=v[1]) for k, v in T.items()})
pairs = ((34, 34), (34, 35), (35, 35), (37, 35), (14, 34))
rows = {f"Kinova3-{l}-from-IIWA-{d}": sum(t + 10 <= len(T["IIWA", d][0]) for t in range(0, len(T["IIWA", d][0]), 5))
        for l, d in pairs}
out["fit_rows_if_every_replay_succeeds_again"] = dict(rows, total=sum(rows.values()), distinct_command_tapes=2)
gap = {}
for l in (34, 35, 36, 37):
    k, i = T["Kinova3", l][0], T["IIWA", l][0]
    n = min(len(k), len(i))
    gap[l] = dict(first5_l2=float(np.linalg.norm(k[:5] - i[:5])), first5_iiwa_norm=float(np.linalg.norm(i[:5])),
                  first_control_differing=int(np.argmax((k[:n] != i[:n]).any(1))))
out["same_layout_same_noise_opening_gap_kinova3_vs_iiwa"] = gap
out["iiwa34_vs_iiwa35_first5_l2"] = float(np.linalg.norm(T["IIWA", 34][0][:5] - T["IIWA", 35][0][:5]))
(HERE / "tape-stats.json").write_text(json.dumps(out, indent=2) + "\n")
