# UR5e π₀.₅ late-oracle handoff

**Ready for parent launch.** Final Baige CPU qualification exited 0 at 2026-11-23 18:46:28 Shanghai on detached revision `cbbbcdca19c7ed3812c3805ce0ee8897109fb9fb`. Worker GPU use was zero. The parent owns the 288 scored episodes. [CPU receipt](evidence/cpu.json) · [Exit and elapsed time](evidence/cpu-exit.txt)

The original 96 queries contain 20 streaming successes, 76 calibrated-retry successes and 82 immediate-oracle successes. The historical retry gained 56 successes and lost zero versus streaming, +58.3 percentage points (95% task-cluster interval +41.7 to +75.0). The original policy executes **five controls per requery** from a ten-action proposal. The reader installs its first estimate after control 10, in time for the motor observation and issued action at **control 11**. Original rows lack per-action evidence for a numerical change at 11 in each historical episode. The fresh run records actual onsets case by case. [Original records and pins](INPUTS.md) · [Qualified query panel](evidence/panel.json.gz)

Run **all 96 matched triplets fresh**, for 288 single-attempt native episodes. The earlier streaming and oracle records lack the opening proposals and first ten issued actions needed to pair them to new outcomes. Each delayed pair must agree bitwise on the complete observed opening, including the state reached after control 10. Every arm shares the original case, native reset, prompt, bank of policy-noise draws and first proposal. Each task starts with an unscored create. A fingerprint rejection can recreate the setup before inference. Every accepted trajectory runs uninterrupted. [Three interventions and estimands](PLAN.md) · [Native rollout](rollout.py) · [Reduction checks](records.py)

## CPU qualification

On final attempt `cpu03`, all eight cases matched the original evaluators at every checked policy query and issued chunk. All eight first applied a nonidentity estimate and first changed the issued action at control 11. The two delayed arms had identical first-ten-control traces. Corrupt noise, post-prefix observation, oracle onset and native return were each rejected. Earlier `cpu01` was superseded during development, and `cpu02` finished the previous validator. The final pin also handles signed-zero no-ops. [Eight task receipts](evidence/cpu.json) · [Linux log](evidence/cpu.log.gz) · [Source hashes](evidence/source.sha256)

The comparison uses the real native UR5e simulator with Mesa software rendering, the frozen seed-41 reader and a deterministic CPU motor whose proposal depends on the policy state and original noise draw. For one nontrivial shift per task, the qualifier compares all motor input, noise, proposal and issued-command hashes with direct calls to the two unmodified historical episode functions. It independently reconstructs the motor observation and inverse action for the delayed oracle at control 11. Seven task checks run through 20 controls. The object-suite task 1 check covers its full 280-control horizon, including the reader's final update and subsequent frozen estimate. [Qualification implementation](qualify.py)

The separate Linux merger test exercises the strict 96-triplet file layout, rejects missing, extra and misidentified cases, and checks positive and negative paired effects on a known task distribution. It includes an inverse gripper command that changes +0.0 to −0.0 to distinguish a numerical action change from a byte difference. Its outcomes are synthetic. The exact-interval implementation also agrees with an independent enumeration on the original 96 simulator-query records. [Merger fixture and result](evidence/merge-fixture.json) · [Independent archive readback](evidence/archive-readback.json)

## Parent submission

Use Baige **分布式训练**, one exclusive node/container, **two RTX PRO 6000D** cards, 75-minute wall limit, automatic retries off. Copy the successful two-card matched-flow template `job-sjubp8q0tg1x` with the project PFS mounted read/write at `/mowenjie`. The qualified UR5e image is `ccr-registry.baidubce.com/aihc/nvcr.io/nvidia/pytorch:2.7.0-cu12.8.0-py3.12-ubuntu24.04` with `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. The launcher keeps the job in that container, probes the entire allocation, checks the CUDA/PCI and CUDA/EGL mappings, then gives each lane one device. No nested scheduler or distributed workers are needed. [Reference runtime](../libero-xemb-ur5e-sysid-20261120-r1/EXECUTION.md) · [Matched-flow launch](../symmeta-matched-flow-20261123-r1/LIVE.md)

Paste this command into the parent’s scheduler job. The detached code is already present on the shared mount. The attempt directory `gpu01` must be fresh.

```bash
bash -lc '
set -eu
run=/mowenjie/fluffy-bassoon-2/outputs/symmeta-late-oracle-20261123-r1
pin=cbbbcdca19c7ed3812c3805ce0ee8897109fb9fb
panel=f23c87c7bbe75d3f5b9b2eb2cfc049541c0c4563162015aa28b4523d974f4c2a
exec bash "$run/code-$pin/launch.sh" run "$pin" gpu01 2 4500 "$panel"
'
```

The measured matched-flow runtime was 1,201 s on two cards for two arms. Adding the measured streaming episode times for the busier lane gives a planning estimate of **1,604 s, about 0.89 GPU-hours**. Exact-arm episode lengths, simulator creates and initialization can move that estimate. The recommended scheduler cap is **2.50 GPU-hours**. For a different single-node allocation, the same executable accepts one to four homogeneous RTX 4090 or RTX PRO 6000D cards. Alternative wall limits that fit this budget are 9,000 s on one card, 3,000 s on three or 2,220 s on four. Change both the scheduler resources and the final card-count and seconds arguments. Use one allocation for the panel. [Measured budget inputs](evidence/budget.json)

The program loads one motor and reader per lane. Lane d takes task shards d, d + card count and so on through shard 7. Each completed query atomically writes its three native records to `gpu01/cases/II.json`. For the two-card layout, check `gpu01/gpu.json`, `gpu01/devices.json`, and cases `00.json` and `12.json` from the two lanes. Each case retains native rewards, return, final simulator success and terminal flags, correction tuple per query, actual noise and proposal hashes, issued command hashes and opening-state receipts. Monitor `lane-0.log`, `lane-1.log` and completed case count every 20 minutes. Completion requires 96 triplets, `summary.json` and status 0 as the first field of `exit.txt`.

The summary gives signed immediate-exact minus delayed-exact (timing), delayed-exact minus streaming (post-opening inference), and immediate-exact minus streaming. It reports wins/losses/ties, percentage-point differences, per-task effects and 95% whole-task intervals for all 96 and for the 64 held-out settings. It also lists fresh-versus-archived discordant indices for the two comparable arms. These first two differences sum to the fresh exact-from-1 versus streaming gap. The historical retry gap includes the extra layout-11 information. The fresh control-11 estimate and per-query trajectory can locate early errors. The delayed-exact advantage includes the effect of later estimate updates and their downstream trajectories.

## Reproduction

The worker snapshot is an independent Git bundle containing the six executable source files. The shared detached checkout and all CPU/GPU attempt directories live under this workstream. The launcher verifies the execution pin, the clean original/recovery/runtime trees, the frozen checkpoints and the CPU-qualified panel SHA-256. The bundle SHA-256 is `58bba5b40d6e20d177dea0a503a2549de69dac4e11628f08e1bdd81756a1677f`. [Worker Git bundle](cbbbcdca19c7ed3812c3805ce0ee8897109fb9fb.bundle) · [Pinned inputs and helper sources](INPUTS.md)

To reproduce the CPU qualification in another fresh directory on `baige-dev`, run the following. The experiment's first real π₀.₅ inference, NVIDIA rendering and scored native episode belong to the parent’s job.

```bash
run=/mowenjie/fluffy-bassoon-2/outputs/symmeta-late-oracle-20261123-r1
pin=cbbbcdca19c7ed3812c3805ce0ee8897109fb9fb
panel=f23c87c7bbe75d3f5b9b2eb2cfc049541c0c4563162015aa28b4523d974f4c2a
bash "$run/code-$pin/launch.sh" cpu "$pin" cpu04 0 2400 "$panel"
```
