# Native retry handoff

The parent pins commits and decides whether a physical offline result warrants GPU time. The registration and case assignment precede the new checkpoints. The source is PR7's verified UR5e π₀.₅ evaluator at ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c. Its 4090 run used prepared UR5e panel SHA-256 `72ea6883cc9eca662041e645c0f31a8e5482f231507b0a1559bdebf44beec6f0`. This supplies the native prompt, embodiment, support, gain/gripper predictions, and contract for the checkpoint and simulator dependencies. This study changes the released query layout and seed. Both 26 and 27 exist in each task's 50-row official state bank. The launcher checks bank bytes and selected release-row fingerprints.

## Artifacts and preparation

Use [cases.csv](cases.csv) for the full preassigned index, task, map, camera signs, reset and noise schedule. The native runner uses [protocol.py](protocol.py) to read the official PR7 panel, frozen policy weight manifest, and simulator sources. [native.py](native.py) runs the prior motor in one process for every arm. An observer records the full 10×7 proposal at each replan, actually issued controls, the physical command received by the native simulator, native reward, predicates, complete reset and opening. The completion summary re-reads all compressed pairs through an independent inverse/forward factor calculation. [Worker observer](audited_worker.py) · [Scoring](score.py)

The reader's gzip JSON exports contain a full-precision forward yaw in degrees, positive forward gain and integer gripper sign. A/B both use seed 41, true calibration, the registered view and matching checkpoint budget. The prediction table also contains wrong-calibration controls and multiple seeds. Joining on index alone would mix experiments. [Export join](export_panel.py)

- Supply the source-only `sealed.json`, physical score receipt(s) produced by the reader, the supporting physical geometry and paired offline evidence. A receipt names its gzip export and SHA, physical bank and SHA, and selected checkpoint seal. Multiple physical receipts may cover disjoint indices. A single scored union bank also works. Include the digital receipt for the registered complement when available.
- Read the physical bank at the actual view for the selected support. Check before/after pixels, three 5-control chunks, identical commands/joint/EEF/jaw across nominal and both orbit views, and the homography. Reconstruct the digital observations independently from the original tape. Join to the rows by original tape, native-view file, observed pixel and supplied H hashes. Verify checkpoint file digests, task, forward truth, gain and sign. Keep the complete matched selected prediction rows with the panel.
- A/B and shipped challenged use their own yaw and the same shipped archival gain/sign. Optional nominal uses the shipped yaw on the nominal view **of that same physical replay**, or of the identical original tape in the digital study. The identity and true-map references use their full stated maps. The parent fixes optional controls before any native reward is observed.
- The exact command options below are for the pinned Baige Linux checkout. `code` is the deployed repository and `pin` is its full commit. `py` is the existing PR7 OpenPI interpreter and `run` is the new run's shared directory. Attempt names must be fresh. All command substitutions use shell-generated local paths supplied by the parent.

~~~sh
study="$code/outputs/symmeta-camera-frame-native-20261124-r1"
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
run="/mowenjie/fluffy-bassoon-2/outputs/${study##*/}"
bash "$study/launch.sh" cpu "$pin" "$cpu_attempt"
~~~

Only after the parent releases native evaluation, seal the approved panel with the actual upstream file paths. Do not infer approval from a file becoming available. Omit the digital argument for a physical-only release. Add a second physical receipt only for disjoint scored banks. Use optional controls selected within budget.

~~~sh
sha=$("$py" -B "$study/release.py" --out "$run/decision.json" \
    --training-seal "$seal" --physical-receipt "$physical_receipt" \
    --geometry-evidence "$geometry" --offline-physics-evidence "$offline" \
    --digital-receipt "$digital_receipt" --controls nominal identity true_map)
bash "$study/launch.sh" prepare "$pin" "$prepare_attempt" "$run/decision.json" "$sha"
~~~

The CPU prepare command verifies the real support/view/checkpoint joins and writes a sealed panel for comparison with the job. The job is one exclusive Baige Distributed Training RTX 4090. Set the scheduler and outer wall time within *remaining* cumulative time, including previous attempts and all probe, startup, rollout and score time. A fresh full-cap run can use 10,740 seconds with a 5-second kill grace and a three-hour allocation ceiling. Shorter allocations must use a correspondingly shorter deadline. The parent owns submission.

~~~sh
timeout --kill-after=5s "${seconds}s" bash "$study/launch.sh" gpu "$pin" "$gpu_attempt" \
    "$run/decision.json" "$sha" "$seconds" "$cpu_attempt" "$prepare_attempt"
~~~

Retain the GPU probe, source pin, launch and exit, sealed panel, per-case traces, and `native-results.json`. A first genuine A/B pair qualifies motor inference and the native reward path in that allocation. A partial run gives evidence on pairing and runtime, with no completed-panel claim. The local record distinguishes CPU fixtures, physical-support retries and digital-support retries. Native here means closed-loop simulation with the released success predicate. No physical robot trial is involved.

The published 4090 run's 384 UR5e π₀.₅ OOD episodes averaged 7.12 seconds inside the recorded motor loop. Across the four complete panels the launcher consumed 14.83 GPU-seconds per episode, including setup and resets. The first number excludes reset and compile overhead. Neither number includes the new per-control audit overhead. [4090 runtime readback](evidence/pr7-4090-runtime.json)
