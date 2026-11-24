# Panda source demonstrations and public execution history

Import `source_data` from this directory. `SourceDataset(root, split="train")` opens a staged dataset. Valid splits are `train` and `heldout`. The latter is reserved for read-only source scoring. `len(ds)` counts examples. `ds[i]` returns a dictionary with the following fields.

| Field | Meaning |
| --- | --- |
| `raw` | Dictionary consumed by stock `to_openpi_observation(raw, prompt)`. Keys are `agentview_image`, `robot0_eye_in_hand_image` (HWC uint8 RGB, native simulator orientation), `robot0_eef_pos` (metric XYZ), `robot0_eef_quat_xyzw`, and `robot0_gripper_qpos` (two finger joints, metres). |
| `prompt` | Native language task description as a string. |
| `actions` | float32 [10, 7], ten consecutive recorded normalized controller commands beginning at the observation. No padding. Apply the chosen stock training action transform downstream. |
| `history` | float32 [52], four oldest-to-newest completed transitions, each with 13 entries as described below. |
| `trajectory_id`, `t` | Stable source trajectory identifier and zero-based observation/action index. |

`PublicHistory()` starts empty. Call `reset()` on every environment reset or state jump. After each real step, call `append(issued_action, previous_raw, next_raw)` with the normalized seven controls actually issued and the public observations on both sides of that step. Call `vector()` before querying the policy. Slots store the seven controls, metric XYZ displacement (3), change in measured jaw width (1), resulting measured jaw width (1), and valid bit (1). Width is abs(finger0 - finger1) in metres. Missing slots are left-padded with 13 zeros. The current observation supplies absolute XYZ. No future controls, rewards, target identities or simulator-only channels enter history. For runtime warmup, reset the buffer at the first scored decision if warmup actions are excluded.

Use the staged `manifest.json` for the exact example/trajectory split and provenance. The final staged path, digest and smoke checks will be recorded in `/tmp/frozen-vla-cotrain-20261124/data-ready.txt`.
