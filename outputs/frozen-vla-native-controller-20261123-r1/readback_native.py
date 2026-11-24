"""Independently read the 24 closed-loop native controller tapes on Linux CPU."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

METHODS = ("OSC_POSE", "hold", "ik", "calibrated")


def inspect(folder):
    summary = json.loads((folder / "live-summary.json").read_text())
    assert not summary["fixture"]
    cells = []
    for layout in (0, 2, 3):
        for suite in ("libero_spatial", "libero_goal"):
            entries = {}
            reset = None
            for method in METHODS:
                file = folder / f"{suite}-{layout}-{method}.json"
                tape = json.loads(file.read_text())
                n = tape["controls"]
                canon = np.array(tape["canonical"])
                native = np.array(tape["native"])
                rewards = np.array(tape["rewards"])
                schema = tape["schema"]
                width = 7 if method == "OSC_POSE" else 8
                assert tape["method"] == method and tape["case"]["layout"] == layout
                assert tape["case"]["suite"] == suite and not tape["fixture"]
                assert tape["meta"]["vendor"] == "NVIDIA Corporation"
                assert n and native.shape == (n,width) and canon.shape == (n,7)
                assert np.isfinite(native).all() and np.isfinite(canon).all()
                assert len(schema["joint_names"]) == 7 and len(schema["low"]) == width
                assert np.all(native >= schema["low"]) and np.all(native <= schema["high"])
                assert np.all(canon[:, -1] == native[:, -1])
                assert rewards.shape == (n,) and np.all(rewards[:-1] == 0) and rewards[-1] == float(tape["success"])
                assert tape["success"] or (tape["truncated"] and n == tape["meta"]["horizon"])
                current = {k:tape[k] for k in ("public","opening","policy_inputs","noise_sha256","meta","case")}
                if reset is None:
                    reset = current
                    assert schema["controller"] == "OSC_POSE" and np.array_equal(native,canon)
                else:
                    assert reset == current and tape["matched_source_success"] == entries["OSC_POSE"]["success"]
                    assert schema["controller"] == "JOINT_POSITION" and schema["native_output"] == [.05]*7
                if method == "hold":
                    assert np.all(native[:,:-1] == 0)
                # Joint commands solved numerically can land just inside bounds. Count exact
                # faces and the source diagnostic's six-decimal reporting precision separately.
                exact = np.any(abs(native[:,:-1]) == 1, axis=1)
                near = np.any(abs(native[:,:-1]) >= 1 - 1e-6, axis=1)
                entries[method] = dict(success=tape["success"], controls=n, reward=float(rewards.sum()),
                    saturated_controls_exact=int(exact.sum()), saturated_controls_within_1e_6=int(near.sum()),
                    peak_normalized_non_gripper=float(abs(native[:,:-1]).max()),
                    raw_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            cells.append(dict(suite=suite, layout=layout, seed=reset["case"]["seed"], task=reset["meta"]["prompt"],
                              horizon=reset["meta"]["horizon"], physics=reset["meta"]["physics"],
                              opening=reset["opening"], noise=reset["noise_sha256"], methods=entries))
    counts = {}
    for m in METHODS:
        seq = [r["methods"][m] for r in cells]
        source = [r["methods"]["OSC_POSE"]["success"] for r in cells]
        counts[m] = dict(cases=len(seq), all_success=sum(x["success"] for x in seq), eligible=sum(source),
                         eligible_success=sum(s and x["success"] for s,x in zip(source,seq)))
        assert counts[m] == summary["counts"][m]
    return dict(method="independent full-tape native reward and action readback", counts=counts, cases=cells)


if __name__ == "__main__":
    print(json.dumps(inspect(Path(sys.argv[1])), indent=2))
