# Native π₀ source smoke

Does nominal π₀ already fail on the source Panda at the development start where we compare it with UR5e? A Panda success falsifies a source failure on this start. A Panda success paired with a UR5e failure localizes the observed difference to the arm comparison on this start. A π₀ failure on both with a π₀.₅ success on Panda is consistent with a source-side policy or interface limitation. Each cell has one predetermined episode. Report 0/1 or 1/1 and avoid estimating a success rate or a general embodiment effect.

Use released BASE development `libero_goal/1` (“put the bowl on the stove”), state 13, seed 20500101. Load each frozen motor once and run Panda then UR5e to the native terminal flag. Use the nominal embodiment and the motor’s identity correction. Both use query noise seed 21317218 and execute five controls per inference, with native H50 for π₀ and H10 for π₀.₅. Compare every same-arm reset field except the horizon-dependent noise-bank digest before inference. Require matching released scene fingerprints and prompt across arms.

**Parent deployment.** Allocate one Baige Distributed Training Pro6K for at most 12 minutes (0.20 GPU-hours). From the new immutable wrapper checkout run the command below with its full commit and a fresh attempt name. The optional third argument identifies the qualified checkout if mounted elsewhere. Its basename must be `source-s2ab-v2` and its HEAD must be `437da7f677632e9495c813aaa52071e207ba834e`.

```bash
bash outputs/xemb-pi0-source-smoke-20261122-r1/run.sh "$(git rev-parse HEAD)" gpu01
```

Under shared storage `outputs/xemb-pi0-source-smoke-20261122-r1/<attempt>/`, collect `result.json`, the hardware probe `gpu.json`, and `exit.txt` (exit code, end and start Unix seconds). The result records native success, return, nonzero rewards indexed by control, motion prefixes, and the first five available inference chunks with full-proposal digests and their first five raw action vectors. Numeric motion traces contain executed controls only. Pairing mismatches and completed cells are saved incrementally. Interpret the panel only with four outcomes, complete status and exit zero. Stop at the first invalid reset, runtime failure or deadline.
