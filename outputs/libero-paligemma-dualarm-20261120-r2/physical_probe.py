"""Compare rendered and camera-free native admission for one pre-registered scene."""
import argparse
import json
from pathlib import Path

import numpy as np

from expert import session
from generated_layouts import native_module, reset_generated
from large_panel_protocol import manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    panel = manifest(root, "dev")
    row = panel["layouts"][0]
    flat = np.load(root / "dev" / row["state_file"], allow_pickle=False)
    native = native_module()
    result = {}
    for arm in ("Panda", "Sawyer"):
        receipts, full, wrists, poses = [], [], [], []
        for camera in (False, True):
            with session(arm, camera) as env:
                obs = reset_generated(env, arm, camera, flat, row)
                runner = env.native.arm if camera else env.arm
                physics = env.reset_info["environment_reproducibility"]
                receipts.append({k: physics[k] for k in row["preflight"][arm]})
                full.append(np.copy(runner._env.get_sim_state()))
                wrists.append(obs["state"].copy())
                poses.append(native.scene_body_poses(runner._env))
        source = row["preflight"][arm]
        result[arm] = dict(camera_free_diff=[k for k in source if receipts[0][k] != source[k]],
                           camera_diff=[k for k in source if receipts[1][k] != source[k]],
                           same_host_full_max=float(np.max(abs(full[1]-full[0]))),
                           same_host_wrist_max=float(np.max(abs(wrists[1]-wrists[0]))),
                           vs_prepared_panda=native.pose_errors(poses[1],row["panda_settled_poses"]),
                           local_pose_error=native.pose_errors(*poses), receipts=receipts)
    Path("diagnostic.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
