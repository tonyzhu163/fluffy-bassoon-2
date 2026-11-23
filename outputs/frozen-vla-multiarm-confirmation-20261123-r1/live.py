"""Run the three predeclared target-only methods through the locked native benchmark."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from native_rollout import raw
from ur5e_motor import load_frozen_pi05
from ur5e_sim_worker import UR5eSimulator
from screen import ARMS, METHODS, MATRIX_SHA, JointChecked, calibrations, cases, check_case, check_pair, historical, key, save


def main(arm, out):
    rows = calibrations(arm)
    history = historical(arm)
    policy = load_frozen_pi05()
    # Match the qualified runner: JAX retains the allocated CUDA GPU. Only the
    # rendering child inherits an empty mask, with an explicitly mapped EGL index.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        simulator = UR5eSimulator()
    with simulator:
        checked = JointChecked(simulator, arm)
        for case in cases(arm):
            stem = key(case)
            row = rows[stem]
            for method, field in METHODS.items():
                matrix = np.asarray(row[field], dtype=np.float32) if field else None
                result = raw(checked, policy, case, matrix)
                check_case(result, case)
                if field is None:
                    baseline = result
                else:
                    check_pair(baseline, result, row, method)
                    result.update(calibration=row, matrices_sha256=MATRIX_SHA)
                result.update(method=method, historical_layout13_raw=history[(case["suite"], case["task_id"])])
                save(out / f"{stem}-{method}.json", result)
                print(stem, method, "native success", result["record"]["success"],
                      "controls", result["record"]["controls"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.arm, args.out)
