"""Verify each issued command against its pinned adapter and establish within-job native outcomes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from adapter import issue


METHODS = ("identity","rank1","rank1_free","smooth_df")


def inspect(folder, matrix_path):
    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    assert matrix_hash == "ad36434d3e748583ed3ca338d26b782fe93aa437652e24abf9013be166edc1a5"
    weights = json.loads(matrix_path.read_text())
    paired = json.loads((folder/"paired.json").read_text())
    assert paired["complete"] and len(paired["cases"]) == 6
    cells = []
    cross = []
    outputs = folder.parents[1]
    for cell in paired["cases"]:
        c = cell["case"]
        key = f"{c['suite']}-{c['task_id']}-{c['init_state_id']}"
        fragment = weights[f"Kinova3/{c['suite']}/{c['task_id']}"]
        methods, reset, opening = {}, None, None
        for method in METHODS:
            f = folder/f"{key}-{method}.json"
            tape = json.loads(f.read_text())
            r = tape["record"]
            assert tape["case"] == c and c["variant"] == "BASE" and c["arm"] == "Kinova3"
            assert not tape["fixture"] and tape["method"] == method and tape["matrices_sha256"] == matrix_hash
            assert tape["support"] == fragment["support"] and tape["support"]["start"] == 11
            assert tape["support"]["controls"] == 15 and tape["support"]["physics"] != r["reset"]["physics"]
            assert c["init_state_id"] in (14,15) and tape["model"]["xml_sha256"]=="2884d72a0427223d333dc00fa2c199c3f86406184c234f436436c7d149002cd7"
            assert r["reset"]["renderer_vendor"]=="NVIDIA Corporation" and tape["model"]["controller"]=="OSC_POSE"
            current = dict(reset=r["reset"],initial=tape["reset_physics"],model=tape["model"],opening=r["opening_proposal_sha256"])
            raw = np.concatenate([p["frozen"] for p in tape["action_packets"]]).astype(np.float32)
            issued = np.concatenate([p["issued"] for p in tape["action_packets"]]).astype(np.float32)
            if reset is None:
                reset, opening = current, raw[:5].copy()
            else:
                assert current==reset and np.array_equal(raw[:5], opening)
            assert len(raw) == len(issued) == r["controls"] == len(r["native_rewards"])
            assert all(x == 0 for x in r["native_rewards"][:-1]) and r["native_rewards"][-1] == float(r["success"])
            assert r["success"] or (r["truncated"] and r["controls"]==r["reset"]["horizon"])
            assert cell["controls"][method] == len(raw) and cell["success"][method] == r["success"]
            conf = fragment["adapters"][method]
            # Preserve the short packet shapes used by the rollout, including its terminal prefix.
            expected = np.concatenate([issue(np.array(packet["frozen"], np.float32), **conf)
                                       for packet in tape["action_packets"]])
            np.testing.assert_array_equal(expected, issued)
            np.testing.assert_array_equal(raw[:,3:], issued[:,3:])
            delta = issued[:,:3].astype(float) - raw[:,:3]
            v = np.array(fragment["adapters"]["rank1"]["direction"])
            off = delta - (delta@v)[:,None]*v
            if method=="rank1":
                assert np.max(np.abs(off)) <= np.finfo(np.float32).eps
            if method=="identity":
                np.testing.assert_array_equal(raw, issued)
            bound = np.any(abs(issued[:,:3])>=1,axis=1)
            new = np.any((abs(issued[:,:3])>=1) & (abs(raw[:,:3])<1),axis=1)
            methods[method] = dict(success=r["success"],controls=r["controls"],
                mean_translation_correction_l2=float(np.linalg.norm(delta,axis=1).mean()),
                maximum_off_axis_change=float(abs(off).max()),issued_boundary_controls=int(bound.sum()),
                newly_saturated_controls=int(new.sum()),raw_file_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        peer={}
        other=outputs/"frozen-vla-native-transfer-20261123-r1"/"gpu01"/f"{key}-Kinova3.json"
        if other.is_file():
            row=json.loads(other.read_text()); prev=row["record"]
            first=np.array(row["chunks"][0]["cmd"],np.float32)
            peer["native_transfer"] = dict(reset_equal=prev["reset"]==reset["reset"],
                differing_reset_fields=[k for k,v in prev["reset"].items() if v != reset["reset"][k]],
                same_motor_input=prev["reset"]["motor_frame0"]==reset["reset"]["motor_frame0"],
                same_noise=prev["reset"]["noise_sha256"]==reset["reset"]["noise_sha256"],
                first_proposal_equal=prev["opening_proposal_sha256"]==reset["opening"],
                opening_max_abs=float(np.max(np.abs(first-opening))),
                opening_rms=float(np.sqrt(np.mean((first-opening)**2))), success=prev["success"],controls=prev["controls"])
        other=outputs/"frozen-vla-visual-adapter-20261123-r1"/"gpu01"/f"raw-{key}.json"
        if other.is_file():
            row=json.loads(other.read_text())
            peer["unpainted_visual"] = dict(reset_equal=row["reset"]==reset["reset"],
                differing_reset_fields=[k for k,v in row["reset"].items() if v != reset["reset"][k]],
                same_motor_input=row["reset"]["motor_frame0"]==reset["reset"]["motor_frame0"],
                same_noise=row["reset"]["noise_sha256"]==reset["reset"]["noise_sha256"],
                first_proposal_equal=row["opening_proposal_sha256"]==reset["opening"],
                success=row["success"],controls=row["controls"])
        cross.append(dict(case=key, raw_kinova=methods["identity"]["success"], comparisons=peer))
        cells.append(dict(case=c,opening=reset["opening"],noise=reset["reset"]["noise_sha256"],
                          physics=reset["reset"]["physics"],methods=methods))
    for label, contrast in paired["contrasts"].items():
        candidate, reference = label.split("_minus_")
        successes = [(r["methods"][candidate]["success"], r["methods"][reference]["success"]) for r in cells]
        counts = {"rescued": successes.count((True,False)), "harmed": successes.count((False,True)),
                  "both_success": successes.count((True,True)), "both_failure": successes.count((False,False))}
        assert all(contrast[name] == count for name, count in counts.items())
        assert contrast["difference"] == (counts["rescued"] - counts["harmed"]) / len(cells)
    return dict(method="independent full-tape action/reward verification",matrix_sha256=matrix_hash,
                cases=cells,raw_repeats=cross,contrasts=paired["contrasts"])


if __name__=="__main__":
    print(json.dumps(inspect(Path(sys.argv[1]), Path(sys.argv[2])),indent=2))
