# Original LIBERO-PRO matched pilot evaluator

**Ready for the parent’s Baige launch.** The locked draw has 32 case–seed pairs (64 one-attempt episodes), eight tasks, four suites and four released perturbation families. The release and historical HF asset pin lack the original Env BDDL/reset pairs, including the requested 8/9 Env cases. Env contributes zero pairs. The case IDs, asset digests, selected state indices, and historical protocol uncertainties are in [the manifest](CASES.json) and [protocol audit](PROTOCOL.md).

## Launch interface

Run from the pinned Linux checkout with the environment provided by [the job wrapper](run.sh). Both the default manifest and a byte-identical copy are accepted. The output directory may already contain scheduler receipts. Its `eval/` subdirectory must be fresh.

```bash
here=$PWD/outputs/symmeta-original-pro-20261123-r1
"$POLICY_PYTHON" -Bu "$here/eval.py" run --manifest "$here/CASES.json" --output "$attempt"
"$POLICY_PYTHON" -Bu "$here/eval.py" report --output "$attempt"
```

The first command evaluates every assigned pair. The second reads the stored contract and available complete-pair records, including after an interruption. It prints assigned and completed counts, missing IDs, and worst/best possible full-panel effects. The scheduler wrapper applies its GPU and wall-time constraints.

For the CPU integration check, use the same runtime paths on allocated Linux with software rendering. The deterministic fixture motor exercises three queries on case zero with the real simulator and reader. Its output is in a fresh `eval-cpu/` subdirectory.

```bash
source "$here/prepare_runtime.sh"
original_pro_paths
original_pro_cpu
original_pro_caches "$scratch"
"$POLICY_PYTHON" -Bu "$here/eval.py" cpu-check --output "$scratch"
```

## Execution and evidence

- **Inputs and resets.** Two fresh simulator processes each create the released scene, call the reference reset, install the assigned variant-specific state, and issue ten settling controls at 20 Hz. The evaluator checks exact state restoration, the scene-selected native Panda and OSC_POSE controller, both 256² cameras, and the declared agentview pose. Both arms use the released filename-to-language function called by the OpenPI client. Each reset receipt also preserves the literal BDDL instruction and its native parser output. An unchanged policy prompt can make semantic perturbations invisible to the motor. The summary counts differences between BDDL and policy instructions after case folding.
- **Motor and reader.** The evaluator loads the frozen official JAX `pi05_libero` policy once and checks the historical checkpoint receipt. Both arms infer with the same explicit float32 diffusion-noise array at each shared query index. Its seed is the big-endian first eight bytes of SHA-256 over UTF-8 `case_id|environment_seed`, plus the historical motor offset 817117. Native sends the policy’s first five actions in their returned precision. SymMeta starts at identity and observes the states reached by its own issued controls. It uses the published seed-41 parameters, DIS features, streaming accumulator, 0.05 prior, two-chunk identity prefix, 24-chunk update limit and observation/action corrections. The reader receives images, end-effector displacements, jaw motion and issued commands. Each new case initializes fresh estimator statistics.
- **Pair validation.** The two arms must match on the reset receipt and every shared prefix query’s motor input, noise, entire proposal, queued actions, resulting public observation, physics, rewards and terminal flag. After ten policy controls, the learned correction can affect the next query. Early native termination counts at the actual control. Both arms use the complete reference horizons, native rewards and success termination. All shared noise indices are checked even after trajectories diverge.
- **Readout.** Each `eval/case-NNN/` contains the full native and SymMeta receipts and their paired result. Query receipts retain estimated yaw, gain and gripper sign, model and applied actions, state correction size, per-control action differences, reader excitation and updates. The summary gives task-, suite- and family-level counts, floor/ceiling, rescues, regressions, interventions, prefix completions, initial successes and duplicated reset states. The paired difference is in success-rate units. The exploratory 95% interval enumerates all 256 within-suite task-cluster bootstrap draws. An exception records the affected case and stops execution. `progress.json` updates after each complete pair. `summary.json` requires all assigned pairs.
- **Linux verification.** The fixture exercised the actual release, checkpointed reader and both camera streams on Baige CPU with Mesa. It reproduced ten prefix controls and three noise indices. At zero-based control 10 the learned adapter changed all five remaining commands. Separate creates exercised the Living Room, Kitchen and Floor scene classes with exact released-state restoration and the native camera poses. The initial integration exposed a draft guard that rejected the released mounted Panda. The final guard follows the released scene’s MountedPanda or OnTheGroundPanda class. Python syntax was checked on macOS. This worker used zero GPU-hours. Actual frozen-policy rewards and success counts depend on the parent’s distributed job.

A separate replay through the pinned historical streaming episode produced all 15 fixture controls byte-for-byte, including the corrected third chunk. Both images and the state vector matched the pinned OpenPI example on all four sampled real observations. A constructed eight-task outcome table recovered four rescues, four regressions and 24 joint failures, the expected task-bootstrap interval and the bounds for 28 missing pairs. The [compact CPU receipt](eval-cpu-evidence.json) includes the exact evaluator file hashes, case, native camera poses, three applied estimates, and shared Linux receipt directory.
