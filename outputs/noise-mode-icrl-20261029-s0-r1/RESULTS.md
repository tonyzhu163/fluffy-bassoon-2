# DSRL noise-arm Stage 0 test results, 2026-10-29

The focused Linux tests pass, but the simulator smoke does **not** qualify a deterministic outcome cache. S0b/C2, S0c/C1 and learner training did not run.

Source: `352edc407738ce14b6a0adb9100732326ca40902`, reviewed from main. Baige job `job-gjkse1sb5gyj`, one NVIDIA RPBZZZ6, 16 CPU, 64 GiB RAM, 8 GiB SHM. Submitted 12:51:14 Shanghai, running 12:51:33, terminal 12:56:24. The scheduler reports Failed/exit 3 because the launcher deliberately refuses admission after the smoke; S0a itself completed normally. No automatic retry.

## Software and runtime checks

- Three focused modules: **31 passed in 8.82 s**, including `tests/test_maniskill_noise.py`. Exact command and output are in [TESTS.md](TESTS.md) and [qualification.log](evidence/qualification.log).
- Allocated CUDA probe passed. SAPIEN rendering verified against allocated PCI `0000:63:00.0`; all 34 environment render devices match.
- Both before/after frozen motor fingerprints equal `ed862a89a39052dad591d6542959ef0a72866b2c85eac2c84767bf55f52205dc`. The native checkpoint, normalization and runtime hashes are recorded in [contract.json](evidence/s0a/contract.json).

## Live smoke

S0a completed **68/68 episodes**, 5,440 task actions and 1,088 motor chunk queries in 228.90 driver seconds. Peak Torch CUDA allocation: 12.31 GiB. Two objects, one scene each, 16 fixed noise arms plus native sampling, two repeats; 34 environments per batch.

| Check | Result |
| --- | --- |
| Fixed-arm complete trajectories identical | **4/32 pairs** |
| Fixed-arm first chunks identical | **5/32 pairs** |
| Success outcome identical | **29/32 pairs** |
| Pairs moved to different slots | **31/32 pairs** |
| First-chunk distance between arms | 0.05155 |
| First-chunk distance between native repeats | 0.03860 |
| Native success | 4/4 |
| Fixed-arm success, pooled across arms | 59/64 |

The three outcome flips are scene 0/arm 3, scene 1/arm 3 and scene 1/arm 9. Each already differs at chunk 0: maximum initial command differences 0.000515, 0.000948 and 0.001342; later maximum differences reach 2.0. Scene IDs, object/overlay/pose indices, arm seeds and arm noise hashes agree within every repeat pair. Every saved trajectory hash and row identity was independently audited; [artifact-audit.json](evidence/artifact-audit.json) gives every pair.

The smoke changes reset seed and batch slot between many repeats. It does not save initial observation hashes, so these artifacts cannot distinguish reset-state/image variation from native numerical effects. This is an empirical repeatability failure of the current collector/runtime combination, not evidence that the noise override is ignored or that learned DSRL cannot work. The tiny nominal smoke gives no held-out headroom estimate.

The all-collection projection was 6,638 s, within the 6,900 s driver budget. Exact repeat determinism failed, so [smoke-admission.json](evidence/smoke-admission.json) stopped the launcher before further GPU collection.

Next: instrument initial observations/state and hold reset seeds fixed when comparing slots. If outcome variation remains, design and budget a repeated Bernoulli cache before evaluating G1–G4. No subsequent launch is implied by this result. Full raw evidence remains at `/mowenjie/fluffy-bassoon-2/outputs/noise-mode-icrl-20261029-s0-r1`; the compact mirror and [manifest](evidence-manifest.json) are retained here.
