"""CPU scoring of the archived source-view proposals against the IIWA donor tapes.

The ten frames are Kinova3 layouts 14 and 15 at controls 0, 30, 45, 60, 90 of the arm's own
closed-loop episode (seed 41810300 + layout), with Panda substituted in the external camera only.
Only control 0 is a state the donor replay also visits; later frames are time-matched to the tape,
not state-matched, and the noise is the Kinova3 episode's, not the donor's.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import teacher

SOURCE = teacher.OUTPUTS / "frozen-vla-source-view-20261123-r1/native-readback"
STEPS = (0, 30, 45, 60, 90)
CONDITIONS = ("raw", "raw_repeat", "control", "panda")


def frames():
    sums = dict(reversed(line.split()) for line in (SOURCE / "GPU01-SHA256SUMS").read_text().splitlines())
    for layout in (14, 15):
        for step in STEPS:
            name = f"proposals-{layout}-{step:03d}.json"
            blob = (SOURCE / "gpu01" / name).read_bytes()
            if hashlib.sha256(blob).hexdigest() != sums[name]:
                raise ValueError(f"{name} differs from the Baige checksum")
            row = json.loads(blob)
            if (row["layout"], row["step"]) != (layout, step):
                raise ValueError(name)
            yield row


def score(tapes=None):
    tapes = tapes or {row: teacher.load_tape(row) for row in (34, 35)}
    rows = []
    for frame in frames():
        boundary = frame["step"] // 5
        row = dict(layout=frame["layout"], control=frame["step"], on_donor_state=frame["step"] == 0, donors={})
        for donor, tape in tapes.items():
            cells = {}
            for name in CONDITIONS:
                five = np.clip(np.asarray(frame["actions"][name], dtype=np.float32)[:5], -1, 1)
                cells[name] = dict(
                    donor={g: teacher.rms(five, tape["sent"][boundary], c) for g, c in teacher.GROUPS.items()},
                    source={g: teacher.rms(five, tape["first_five"][boundary], c) for g, c in teacher.GROUPS.items()})
            row["donors"][str(donor)] = cells
        row["panda_minus_raw_rms"] = teacher.rms(
            np.clip(frame["actions"]["panda"], -1, 1), np.clip(frame["actions"]["raw"], -1, 1))
        rows.append(row)
    summary = {}
    for donor in map(str, tapes):
        for scope, keep in (("all_frames", lambda row: True), ("control_0", lambda row: row["on_donor_state"])):
            for target in ("donor", "source"):
                for group in teacher.GROUPS:
                    raw, paired = {}, {}
                    for row in filter(keep, rows):
                        cells = row["donors"][donor]
                        raw.setdefault(row["layout"], []).append(cells["raw"][target][group])
                        paired.setdefault(row["layout"], []).append(
                            cells["panda"][target][group] - cells["raw"][target][group])
                    summary[f"iiwa{donor}/{scope}/{target}/{group}"] = dict(
                        raw=teacher.layout_interval(raw), panda_minus_raw=teacher.layout_interval(paired),
                        frames_closer=int(sum(v < 0 for values in paired.values() for v in values)),
                        frames=int(sum(map(len, paired.values()))))
    return dict(evidence="archived replay: GPU forward passes of job-69hluhi5wgb7 rescored on CPU; no reward",
                frames=rows, summary=summary)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "evidence"
    out.mkdir(exist_ok=True)
    report = score()
    teacher.write_json(out / "archived.json", report)
    for key, cell in report["summary"].items():
        if key.endswith("/all") and "/donor/" in key:
            print(key, "raw %.4f" % cell["raw"]["mean"], "panda-raw %+.4f" % cell["panda_minus_raw"]["mean"],
                  cell["panda_minus_raw"]["layout_interval95"], cell["panda_minus_raw"]["by_layout"],
                  f'{cell["frames_closer"]}/{cell["frames"]}')
