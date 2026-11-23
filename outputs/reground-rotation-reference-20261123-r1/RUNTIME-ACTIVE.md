# Runtime handoff

**CPU ready. Development-GPU smoke in progress.** The Baige shared root is `/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1`. In Bash, source `$root/runtime/activate.sh` and use `$root/env/bin/python`. The release is `$root/upstream` at `16a98486f560676b9036d1a6d067b9dd51a5917a`.

All 150 packages installed, `uv pip check` passed, and `python main.py setup` generated `$root/libero-config/config.yaml`. The CPU reference smoke passed a reset, both 256 × 256 views and one step on `spatial_0`, `rot_z/90`. See `$root/cpu-smoke.json`. The RTX 4090 smoke started at 15:06:43 Shanghai on `mwj-clean`. Its log is `$root/dev01.log`.

The run-local Hub cache at `$root/hf/hub` holds the byte-verified nine-file policy, six tokenizer files, and the exact PaliGemma config that Transformers 5.5.4 requires. The model pin is `7152f13e34c6c115d808aea0e5d08aca74eac14b` and the tokenizer pin is `35e4f46485b4d07967e7e9935bc3786aad50687c`. Use the released `VLA('pi05', 'cuda', revision='7152f13e34c6c115d808aea0e5d08aca74eac14b')`. Unversioned native calls resolve the same pin through the offline metadata hook. All 586 reference asset files are at `$root/assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`. Receipts are `$root/hf-verified.jsonl` and `$root/assets-verified.json`.
