# Calibrated registration helped on physical camera orbits

On 64 held-out UR5e supports, expressing observed motion in the calibrated reference frame reduced yaw error from **16.52° to 8.39°** on real ±30° camera orbits. The paired A−B difference was **−8.13°**, with a 95% whole-task interval of **−10.78° to −5.35°**. Each of the eight tasks and all three seeds favored registration. On the preregistered digital ±20°/±45° roll comparison, A averaged **6.34°**, B **6.10°**, and the difference was **+0.24° [−0.35°, +0.84°]**. This changes our prior in two directions. The strong raw-view baseline handled moderate roll well. Physical changes in viewpoint still benefited from explicit planar alignment. [Preregistration and before-fit amendments](PLAN.md) · [Paired estimates and control strata](evidence/contrasts.json)

The original failure was a 5.83° upright error rising to 85–90° when only the shipped reader's camera view was rolled a quarter turn. The test here changes camera view independently of the command frame. Each target recording retains its actual native yaw, gain, gripper polarity, commands and reported displacement. The frozen π₀.₅ motor always used its original view. The endpoint is offline command-map estimation. A separate workstream owns native retries. [Original input interventions](../symmeta-perceiver-mechanism-20261124-r1/RESULT.md) · [Native evaluation registration](../symmeta-camera-frame-native-20261124-r1/PLAN.md)

## Matched comparison

Both readers have 2,124,438 parameters. Each receives three chronological 64×64 RGB pairs, three issued 7D commands, reported 3D end-effector and 2D jaw increments, chunk validity, and the same known reference-to-observed homography. A samples **the actual observed RGB** back onto the reference grid and recomputes DIS there. B computes DIS on observed RGB. Before spatial pooling, both CNNs receive the nine PR7 channels, the two coordinates of the corresponding point in the other grid, and an overlap bit. Both receive nine normalized H coefficients again on each 117-dimensional Perceiver token. B therefore has access to pixel-level correspondence while it learns its motion features. A can lose detail at the second resampling and cannot recover an occluded surface. [Observation pipeline](camera.py) · [Actual token and CNN inputs](model.py) · [Linux fixture and reference loss check](evidence/qualification.json)

We verified all eight PR7 source shards and used all 1,466 Panda/Sawyer tapes. Both arms saw identical source samples, independent camera draws and continuously relabeled native command interfaces at every update, with seeds 41/42/43 and batch 32. The source bank contained upright, rolls of both signs at 12°, 35°, 60° and 80°, synthetic ±18° planar orbits, combinations with roll and off-axis crops. Camera variation covered both translation axes. Digital source warps can teach image correspondences and introduce pose-specific padding. Physical rendering additionally changes visibility and depth-dependent motion. [Source loader](data.py) · [Source input and mask audit](../symmeta-camera-frame-review-20261124-r1/source-inputs.json)

The PR7 schedule used a 3e−4 peak rate, 200-update warmup and 2,500 updates. We gave both arms a second matched 2,500-update cycle with the optimizer moments retained. On 224 separate source recordings with no exact four-frame duplicate in training, B's mean error across 11 registered validation poses fell from 10.68° to 6.18°, selecting the 5,000-update checkpoint for both arms before opening target scores. At 5,000, A averaged 5.15° and B 6.18°. On upright source validation they reached 3.70° and 4.29°. The original shipped reader scored 4.32° on those same upright inputs. On eight separately rendered source tapes from one task, the sealed A/B fits scored 5.99°/8.90° at −15° physical orbit and 8.19°/10.09° at +15°. B was still improving at the end, from 7.00° at 4,500. The comparison supports a finite-budget generalization effect. The attainable physical error under longer training or authentic source orbits remains open. [Per-seed source curves](evidence/source-curves.json) · [Checkpoint selection and hashes](evidence/sealed.json) · [Independent replay of training draws](../symmeta-camera-frame-review-20261124-r1/fit-readback.json) · [Source factor audit](../symmeta-camera-frame-review-20261124-r1/source-factors.json) · [Physical source tapes](../symmeta-camera-frame-review-20261124-r1/source-orbit.json)

## Held-out camera and control combinations

All errors below are mean absolute circular yaw error in degrees, lower is better. Brackets contain paired 95% percentile intervals from 20,000 resamples of whole tasks. The estimator averages repeated views and seeds within task. The 64 OOD recordings comprise four compositional and four continuous interfaces on each of eight tasks. All 96 official supports were scored digitally. The physical bank covers all 64 OOD identities; each set of moved views was rendered at identical instants along its own fixed replay trajectory. Recorded five-command means were re-executed for those replays. The physical nominal row supplies their matched reference. [Physical bank and trajectory validation](../symmeta-camera-pose-pairs-20261124-r1/RESULT.md)

| Support camera | A registered | B raw + spatial H | A−B [95% interval] | Shipped raw | Shipped yaw + known roll |
|---|---:|---:|---:|---:|---:|
| Original digital nominal | 6.30 | 6.49 | −0.19 [−0.73, +0.30] | 5.82 | 5.82 |
| Digital −45° | 6.32 | 6.74 | −0.42 [−1.03, +0.14] | 53.47 | 22.18 |
| Digital −20° | 6.37 | 6.41 | −0.04 [−0.50, +0.48] | 25.06 | 12.00 |
| Digital +20° | 6.34 | 5.42 | +0.92 [+0.07, +1.80] | 23.06 | 10.86 |
| Digital +45° | 6.34 | 5.81 | +0.53 [−0.14, +1.20] | 51.33 | 19.67 |
| Digital quarter turn −90° | 6.01 | 11.36 | −5.35 [−7.03, −3.66] | 84.71 | 26.15 |
| Digital quarter turn +90° | 6.05 | 11.71 | −5.66 [−6.59, −4.79] | 90.12 | 31.71 |
| Digital plane orbit −30° | 6.87 | 6.55 | +0.33 [−0.78, +1.56] | 24.25 | — |
| Digital plane orbit +30° | 6.94 | 7.38 | −0.44 [−0.93, +0.10] | 30.76 | — |
| Physical nominal | 6.10 | 6.04 | +0.06 [−0.62, +0.74] | 5.82 | 5.82 |
| Physical orbit −30° | 8.05 | 15.17 | −7.12 [−9.36, −4.49] | 32.45 | — |
| Physical orbit +30° | 8.74 | 17.87 | −9.13 [−12.83, −5.86] | 22.86 | — |
| Physical optical roll −20° | 6.40 | 5.49 | +0.91 [−0.10, +2.01] | 25.42 | 11.70 |
| Physical optical roll +20° | 6.34 | 5.38 | +0.95 [−0.18, +2.10] | 23.20 | 9.08 |

On digital moderate rolls the seed-level A−B differences were +2.18°, −1.04° and −0.41° (sample SD 1.71°). On physical orbits they were −5.85°, −6.24° and −12.30° (SD 3.62°). The physical A−B difference was −7.28° [−10.97°, −3.89°] on the 32 compositional cases and −8.98° [−11.40°, −6.40°] on the 32 continuous cases. The original eight cases on two tasks gave −10.98° [−12.90°, −9.07°]; the additional 56 gave −7.84° [−10.41°, −5.25°]. The two-task interval describes only the original pair of tasks. Per-support deltas, native control IDs, signed factors, both gain and gripper errors, task means and all three seed estimates are retained. [Digital scores](evidence/fit01/digital-summary.json) · [Physical scores](evidence/phyGpu01/physical-summary.json) · [Native control strata](evidence/contrasts.json)

## What the controls establish

- **Upright cost.** Against shipped upright on the original 64 supports, A changed error by +0.48° [−0.42°, +1.23°] and B by +0.67° [−0.01°, +1.38°]. Under physical orbit, relative to each model's own same-trajectory nominal estimate, A incurred +2.30° [+0.05°, +4.50°] and B +10.49° [+7.79°, +13.23°]. The known-map oracle has zero yaw error by construction. Nominal reader errors measure the cost relative to an observed reference view. [Paired costs](evidence/contrasts.json)
- **Calibration matters.** Giving both models the opposite camera calibration while retaining identical physical observations increased orbit error by 50.58° [48.33°, 52.65°] for A and 26.21° [24.58°, 27.89°] for B. On moderate digital rolls the increases were 60.76° and 16.48°. B learned to use supplied calibration. Its physical orbit deficit remained with the correct calibration. [Per-view interventions](evidence/phyGpu01/physical-summary.json)
- **Exact restoration.** With the inverse quarter turn and identity metadata, all digital RGB pairs, DIS and nine PR7 features matched upright, and every A output factor on all 96 tapes and three seeds matched upright exactly. The conditioned quarter-turn rows in the table retain the real H and can differ from upright. At a lossy angle, editing the observed image changed registered RGB and motion. Both networks' added spatial channels received gradients in the source fixture. [Qualification](evidence/qualification.json) · [Spatial gradient check](evidence/local-gradient.json)
- **How much needs new weights?** A separate matched replay applied the same public geometric correction before the three original frozen shipped readers. Physical orbit error fell from 27.65° raw to 9.59° with zero updates. A added a smaller 1.19° reduction [0.41°, 2.06°] relative to the frozen correction, with the gain concentrated on +30°. On moderate digital rolls, frozen correction reached 9.28°, compared with 6.34° for A and 6.10° for B. Most of the physical rescue can therefore be obtained by coordinate normalization before an existing reader. [Matched physical replay](../symmeta-camera-frozen-preprocess-20261124-r1/evidence/physical-join/joined-summary.json) · [Matched digital replay](../symmeta-camera-frozen-preprocess-20261124-r1/evidence/digital-join/joined-summary.json)

## What augmentation alone learned

Before any target scores were available, we registered a third source-trained reader C. It sees B’s observed RGB, DIS, validity and commands. We clamp both calibration inputs to identity throughout training and evaluation. C retains visual orientation clues and the synthetic padding pattern. Its exposure hashes match B for all three seeds, and it uses B’s source-selected budget. At 5,000 updates its source-validation MAE was 7.86°. [Augmentation-only registration](AUGMENTATION-CONTROL.md) · [C exposure and checkpoint seal](evidence/blind01/sealed.json)

| Held-out camera | C augmentation alone | B augmentation + H | C−B [95% task interval] |
|---|---:|---:|---:|
| Digital ±20°/±45° | 6.77° | 6.10° | +0.67° [+0.01°, +1.33°] |
| Physical optical roll −20° | 8.85° | 5.49° | +3.36° [+2.30°, +4.56°] |
| Physical optical roll +20° | 7.23° | 5.38° | +1.85° [+1.13°, +2.51°] |
| Physical orbit −30° | 30.52° | 15.17° | +15.35° [+13.26°, +17.08°] |
| Physical orbit +30° | 22.87° | 17.87° | +5.00° [+2.18°, +7.54°] |
| Physical orbits pooled | 26.69° | 16.52° | +10.17° [+8.70°, +11.72°] |

C preserved most of the moderate digital-roll improvement over the shipped raw reader. Under physical orbit, explicit calibration in B reduced C’s error by 10.17°, and applying that calibration geometrically in A reduced B’s error by a further 8.13°. The opposite-H intervention and the separately trained C point in the same direction. They leave the contribution of synthetic padding versus ordinary scene orientation unresolved. [Digital C results](evidence/blind01/digital-C-summary.json) · [Physical C results](evidence/blind01/physical-C-summary.json) · [Pooled paired contrast](evidence/blind01/pooled-physical.json)

The predeclared reset-TCP plane reduced mean tool-endpoint registration error from about 2.1 pixels to 0.66 pixels on the real orbits. Whole-image RGB agreement had mixed changes across tasks. This makes near-plane motion alignment a plausible explanation for A's physical advantage. Scene depth, occlusion, the source padding cue, and optimization all remain possible contributors. The unchanged policy and offline endpoint leave task reward to the separately registered native evaluation. [Physical geometry readback](../symmeta-camera-pose-pairs-20261124-r1/RESULT.md)

## Reproduction and handoff

Baige shared root is `/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-frame-reader-20261124-r1/`. Inputs are the fingerprinted PR7 tapes, the omitted-chunk source-only validation set and the pinned physical bank. The source-only selection seal stores each checkpoint path basename, SHA-256, model config and exposure hash. Use `model.restore(path, seed)` with the camera preprocessing in this run, and apply the forward factors yaw degrees, positive XY gain and gripper sign ±1 using the native command adapter. Target truth, task and transform IDs are scoring metadata. [Sealed deployment manifest](evidence/sealed.json) · [Digital rows](evidence/fit01/digital-predictions.json.gz) · [Physical rows](evidence/phyGpu01/physical-predictions.json.gz)

| Stage | Code pin | Attempt or job | Wall time |
|---|---|---|---:|
| CPU source bank, cores 34/35 | `8aab9cf0a7188f183ea7dcdc507858df515efaa0` | `qual01`, `prep01` | 209 s |
| CPU final qualification and off-axis supplement, cores 34/35 | `68cedbfe546836ac6fbb8ec6129192d2deb58f94` | `qual02`, `offset01` | 121 s |
| Matched fits and digital scoring, one RTX 4090 | `68cedbfe546836ac6fbb8ec6129192d2deb58f94` | `job-gkw4h79jzonj`, `fit01` | 384 s |
| Physical scoring, one RTX 4090 | `7022a39ab0384f515a1930eaa81e180eab829bf2` | `job-ggya6d8bcpre`, `phyGpu01` | 142 s |
| Augmentation-only fit and both offline panels, one RTX 4090 | `ed719f21e7f5f27ce629c00eed80f90caeb9d48b` | `job-8ihblh9iya5f`, `blind01` | 306 s |

All three GPU launchers exited 0, using **0.2311 GPU-hours** in total. The main A/B operations accounted for 0.1461 GPU-hours. All twelve trained A/B parameter files at 2,500 and 5,000 updates remain in `fit01/`. All six C parameter files remain in `blind01/` with the same naming convention and prefix C. The selected A/B files are `{A,B}-seed{41,42,43}-step5000.msgpack`. The 14,976 digital rows and 4,992 physical rows include each reader's factors at full precision and the observed-image, tape, H and checkpoint digests. [Digital receipt](evidence/fit01/digital-receipt.json) · [Physical receipt](evidence/phyGpu01/physical-receipt.json) · [C digital receipt](evidence/blind01/digital-C-receipt.json) · [C physical receipt](evidence/blind01/physical-C-receipt.json)

Run the commands below from their respective pinned checkout roots with fresh attempt names for reproduction. The named attempts reproduce the recorded path structure. The CPU modes run on the allocated cores. The GPU modes require an exclusive Baige Distributed Training job.

```sh
r=outputs/symmeta-camera-frame-reader-20261124-r1/run.sh
bash "$r" prepare 8aab9cf0a7188f183ea7dcdc507858df515efaa0 prep01
bash "$r" qualify 68cedbfe546836ac6fbb8ec6129192d2deb58f94 qual02
bash "$r" extra 68cedbfe546836ac6fbb8ec6129192d2deb58f94 offset01 prep01
bash "$r" fit 68cedbfe546836ac6fbb8ec6129192d2deb58f94 fit01 prep01 offset01
bank=/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-pose-pairs-20261124-r1/cpu01/ood64/bank.json
bash "$r" physical-gpu 7022a39ab0384f515a1930eaa81e180eab829bf2 phyGpu01 fit01 "$bank"
bash "$r" blind ed719f21e7f5f27ce629c00eed80f90caeb9d48b blind01 prep01 offset01 fit01 phyGpu01 "$bank"
```
