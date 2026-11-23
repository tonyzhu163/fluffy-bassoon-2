# Supported correction on genuine native-arm traces

**Decision for the parent.** Hard truncation did not consistently improve prediction over equal-DF ridge on the 24 genuine target supports. Restricting the issued correction changed 15-control Kinova3 endpoints by **1.94–7.39 mm** on six paired starts. Removing the restriction from the same rank-one forward fit brought all six endpoints within **0.11 mm** of matched ridge. Relative to raw, the supported correction increased source-motion error on 2/6 starts and matched ridge on 1/6. The parent's submission receipt shows Baige job `job-495gtsfh6x6y` queued on one GPU. The command below records that comparison. This worker used zero GPU-hours. [Final table and limitations](RESULT.md) · [Measured physical effects](evidence/replay-summary.json)

**First Baige CPU result.** With λ = 1 from the sibling's Panda/Sawyer selection, top-direction ridge versus full smooth ridge gave 3.091469 versus 3.055846 mm on Kinova3, 2.387863 versus 2.390823 mm on IIWA, and 3.331639 versus 3.343218 mm on UR5e. Each is a mean over 56 cross-task support/query pairs from eight tasks on that arm. This first screen shows no consistent gain from truncating the forward model. The command comparison tests a different consequence of the architecture.

Exact command from the repository root, one pinned Baige CPU core:

```sh
ssh baige-dev bash -s < outputs/frozen-vla-supported-correction-20261123-r1/initial.sh
```

[Measured stdout](initial.log) · [Plan](PLAN.md)

**Matched filter readout.** Source-only tuning selected λ=1 for the leading direction and 0.1 for the orthogonal control. Rank one minus smooth ridge at identical per-output effective degrees of freedom was +0.025 mm [−0.050, +0.100] on Kinova3, −0.004 [−0.026, +0.017] on IIWA, and −0.018 [−0.041, +0.006] on UR5e (approximate 95% delete-one-task intervals). On Kinova3 the unrestricted matched-ridge inverse added 0.01468 normalized command units off the leading axis, averaged over held-out five-control means. The supported inverse preserved that component to float32 precision. [24 fitted adapter records](evidence/adapters.json) · [Statistics](evidence/summary.json)

Full CPU command from the repository root on Baige is below. The immutable source checkout is inside the owned run. Use a fresh result directory for a repeat.

```sh
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-supported-correction-20261123-r1
bash "$run/src-cpu/cpu.sh" "$run/cpu-repeat"
```

**One Kinova3 native pilot passed CPU qualification.** The first four native-physics replays put the constrained and equal-DF ridge endpoints 1.94–2.88 mm apart after 15 identical input controls. On cream-cheese layout 14, the supported inverse increased five-control mean source-motion error from 4.825 to 4.955 mm, whereas smooth ridge reduced it to 3.878 mm. This makes over-restricting the correction a concrete risk. Native task reward remains unscored for these new adapters. A CPU fixture exercised all four branches using one recorded policy opening and verified paired physics, opening actions, wrist/gripper and off-axis preservation. [Fixture](evidence/fixture.json)

The parent submitted one Baige Distributed Training job. Its receipt shows one queued GPU. The qualified command follows for provenance, with one RTX 4090 or RTX PRO 6000D and a scheduler wall limit of 1 h. [Submission receipt](submission.jpg) The detached checkout and input already exist on shared storage. The launcher runs six frozen-policy starts × four methods and uses a fresh attempt directory. No submission by this worker.

```sh
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-supported-correction-20261123-r1
bash "$run/src-native/native.sh" d8d7b63c58d1cf37132e6bfa3582e24258005027 gpu01
```

Matrices are at `$run/cpu01/adapters.json`, SHA-256 `ad36434d3e748583ed3ca338d26b782fe93aa437652e24abf9013be166edc1a5`. Use the stored direction and the scalar box limiter for supported rows. A plain matrix multiply followed by coordinate clipping can violate the intervention. [Plan](PLAN.md) · [Results](RESULT.md) · [Paired launcher](native.sh)
