# Native transfer on Panda fitting tasks

The lexicographic rule selected alphabet soup into the basket (Object 0) and the black bowl between the plate and ramekin onto the plate (Spatial 0). For each task the staged Panda split contains 24 fitting windows and eight validation windows. The native instruction uses “pick the alphabet soup” for Object and “pick the akita black bowl” for Spatial. In each case the source prompt matches the released BDDL **filename**, and both evaluated robots receive the same native BDDL instruction. The robot change therefore holds the test wording fixed. [Preregistration](PLAN.md) · [Source-to-native mapping, episode IDs, prompts and geometry](contract.json)

| Task | Panda train episode → original demonstration | Read-only validation episode → original demonstration | Released rows | Reset seeds on both arms | Horizon |
| --- | --- | --- | --- | --- | ---: |
| Object 0 | 813 → 30, 818 → 12 | 825 → 20 | 48, 49 | 42820048, 42820049 | 280 |
| Spatial 0 | 1272 → 15, 1273 → 46, 1275 → 11 | 1282 → 1 | 48, 49 | 42830048, 42830049 | 220 |

Each recorded action window in the staged HDF5 was compared bitwise with the corresponding seven-dimensional float32 actions at the same indices in every same-length original Panda demonstration, after the upstream no-op filter. The matches above are unique. For every released row and every matched source demonstration, the contract records a named free object's XY positions in metres. The smallest, across demonstrations, of the largest object displacement was 7.28 and 9.88 mm on Object rows 48 and 49, and 21.89 and 25.06 mm on Spatial rows 48 and 49. The native scene-transfer code admits at most 1 mm of position error. Coordinates use the original demo XML ordering and the native compiled joint addresses, with time, robot joints and velocities excluded. [Original archive revision](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a) · [Replay construction](https://github.com/openvla/openvla/blob/0ebc5e333ee7a916813bf50db240f1af7ba1d295/experiments/robot/libero/regenerate_libero_dataset.py) · [Registration code](register.py)

## Parent launch

One Baige Distributed Training replica, one RTX 4090 or RTX PRO 6000D, 12 CPU cores, 64 GiB, automatic retries zero, scheduler limit **90 minutes**. The parent owns the 12 GPU-hour aggregate ledger and submission. The shell bounds the process to 5,300 seconds plus five seconds to terminate. The command below uses the recovery attempt `gpu02` if it successfully publishes the same fit used by the main panel. For a later successful recovery, replace only the final argument with its attempt name. The launcher checks the training exit receipt and the independent digest printed at publication. It resolves the model and history implementation from the detached training revision recorded in that fit.

```bash
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1
pin=c51e767770bc48827a1d6d3837cddefe1b7a1d6c
launcher="$run/matched-task/code-$pin/outputs/frozen-vla-cotraining-20261124-r1/matched-task/launch.sh"
bash "$launcher" gpu "$pin" gpu01 gpu02
```

The launcher loads one locked model. It verifies the staged manifest `05209732b5e3f04308c8acbef2f6c8c8a2f173673840fdb055e1b094ff7d6f40`, source window identities, fitting draws and complete epoch coverage before scoring. Both native arms use nominal OSC, the released camera views and five controls per replan. GPU mode uses the parent's public history, parity checks, inference implementation, native episode engine and trajectory validator. The Object arena requires the release's on-the-ground mount. Spatial uses the tabletop mount. The evaluator verifies each mount explicitly.

Output is `$run/matched-task/gpu01/`. `episodes/` contains all 40 terminal reward, action, history and reset tapes. `native-summary.json` requires all 40, matching released scenes/noise across arms, matching initial observations within arm and unchanged parameter/normalization hashes. It contains all eight arm/start cases and the four crossed differences, rescues, losses and leave-one-start-out estimates. The unit for the crossed contrast is one task/row. The four starts support a descriptive comparison on these two familiar tasks. The stock policy's pretraining scene exposure is unknown.

## CPU qualification

Baige CPU cores 42 and 43 ran all 40 native fixture episodes to the 15-control fixture limit. All within-arm opening states, images, inputs and noise reproduced on the first admitted attempt. The four crossed scene and diffusion-bank comparisons matched. The maximum post-settle discrepancy across the native scene bodies was 1.01 × 10⁻¹⁶ m in position and 3.43 × 10⁻¹⁶ rad in rotation (rounded up). On all eight starts the deterministic fixture's joint and history-zero paths first issued different commands at zero-based control 5. The fixture weight digest was unchanged. Separate stationary CPU episodes on the same Panda tasks reached the official 280- and 220-control timeouts with the native truncation flag. These checks exercise the simulator, intervention plumbing and stopping rules. Fitted-policy outcomes await the parent's job. [CPU readback](evidence/cpu-readback.json) · [Forty complete fixture tapes](evidence/cpu02-fixture.tar.gz) · [Uncapped native timeouts](evidence/terminal-cpu.json)

The first fixture attempt used the parent panel's tabletop admission check and stopped before issuing its first scored control in the Object floor arena. Attempt `cpu02` explicitly checks the native floor and tabletop mount classes and reuses the same public history and native rollout engine. Its executable was `7bb692deb31f675a362f3a0af0f770e9ce24082a`. The terminal smoke used `29f53e978e17628898b96d452699a7ebae57398b`. The handoff pin above read back and independently validated all 40 archived episodes. It also admits the direct-fit recovery's per-example receipt. The worker submitted zero GPU jobs and used zero GPU-hours.
