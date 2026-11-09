# Released Motus π0.5 phone-stand baseline

The frozen released checkpoint measured **27/100 clean and25/100 randomized**
successes on `place_phone_stand`. This establishes a baseline for the documented
native/OpenPI reconstruction; it does **not** reproduce Motus Table14's49/53.
It is one task, not the paper's50-task aggregate.

| Domain | Successes | Rate | Wilson95% interval | Paper |
|---|---:|---:|---:|---:|
| Clean |27/100|27%|19.27–36.43%|49%|
| Randomized |25/100|25%|17.55–34.30%|53%|

Reference: [Motus v2, Table14](https://arxiv.org/html/2512.13030v2).
Checkpoint: [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2)
at `effc6e1a99bd4d0edf9447e8ba313def67649cfe`.

## Protocol and evidence

- Source `e15b1ac9805c84b9b4f82d489cdde0eaf8d86e57`, native RoboTwin
  `fa8d97b0f33bb894eae16df5a8175490f5b977a7`; direct reference OpenPI,
  H32/K10 eager ODE, measured14D joint state, unseen native instructions,
  original `Base_Task.take_action`,400-target episode limit.
- Native seed index0, candidates starting100000. Expert feasibility admits
  the first100 successful scenes per domain. All36 clean and71 randomized
  rejections are retained. Candidate ranges end100135/100170 without gaps.
  No policy failures were excluded.
- Independent Linux CPU [readback](PHONE100-READBACK.json) passed all200
  episodes,200 lossless initial camera/state/prompt inputs,2,161 finite32x14
  action chunks,66,062 delivered targets, frozen parameter version checks,
  zero optimizer updates,507 allocated-GPU renderer receipts and zero denoiser
  errors. Runtime, source, model, norm and asset receipts match qualification.
- Phone, stand and ALOHA robot assets: all120 upstream files match the dated
  official dataset ([audit](UPSTREAM-ASSET-AUDIT.json)). Complete1,000 unseen
  textures are verified. Private OIDN2.3.3 fixes Blackwell support while keeping
  native SAPIEN physics/rendering and shader settings; shared runtime untouched.
- [Raw result](PHONE100-RESULT.json), [GPU probe](PHONE100-GPU-PROBE.json),
  [exit](PHONE100-EXIT.json), [frozen evaluation cases](PHONE100-CASES.json).
  Full images/action files remain in BOS under the recorded execution output.

## Interpretation and next step

The exact released π0.5 evaluation fork/config remains unavailable in the
reviewed Motus release. Its metadata references an unreleased
`add_state_from_actions` training transform. K10 is the reference OpenPI
default, not a maintainer-confirmed paper setting. Asset/runtime checks rule
out those identified infrastructure faults, but do not identify the cause of
the22/28 percentage-point gap. Do not describe this result as reproducing the
paper, or tune on these evaluation outcomes.

Freeze these200 cases and prompts for subsequent comparisons. Qualify the
RLinf starting policy's preprocessing and action output against the direct
reference before launching distinct PPO and GRPO baselines. The prior custom
waypoint recipe is not this native controller and cannot be resumed unchanged.

## Cost and closeout

Baige Pro6K `job-l12f2zavadif` succeeded/released, created2026-11-09 13:55:27
Shanghai, ended15:59:51. Worker7,443GPU-s; conservative creation-to-terminal
charge7,464GPU-s (2.0733GPU-h). Including all earlier Epoch2 work:9,638GPU-s
(2.6772/100GPU-h);97.3228GPU-h remain. No active job or PPO/GRPO training at
this closeout. See the single [budget ledger](../robotwin-restoration-20261106-r1/LATEST-BUDGET.json).
