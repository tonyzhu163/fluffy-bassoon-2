# Native task outcomes on IIWA and UR5e

**Full 3D and planar translation corrections produced identical native success outcomes on every task.** IIWA solved the same 7/8 tasks under raw, planar and full 3D commands. UR5e solved 7/8 raw and 6/8 under each correction. Both corrections lost the UR5e two-moka-pots task and rescued zero raw failures on either arm. All 48 registered simulation episodes completed. These results give no reason to prefer full 3D over planar for native success on this screen. [Registered design](PLAN.md) · [Production readback and receipt hashes](evidence/production-native.json)

## Native task table

Each cell gives controls to the first unit of native reward. **F** marks a failure, with controls to the released horizon in parentheses. Each row is the same held-out task on BASE layout 14 for both arms. The evidence includes the released language instruction, seed, terminal flags and return for every episode. [Episode records](evidence/production-native.json)

| Released task | IIWA raw | IIWA planar | IIWA 3D | UR5e raw | UR5e planar | UR5e 3D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `libero_10/5` | 183 | 180 | 184 | 182 | 181 | 183 |
| `libero_10/8` | 353 | 371 | 327 | 470 | F (520) | F (520) |
| `libero_goal/3` | F (300) | F (300) | F (300) | F (300) | F (300) | F (300) |
| `libero_goal/6` | 114 | 81 | 113 | 123 | 122 | 120 |
| `libero_object/1` | 133 | 138 | 143 | 129 | 140 | 147 |
| `libero_object/8` | 143 | 143 | 143 | 149 | 154 | 153 |
| `libero_spatial/0` | 183 | 81 | 81 | 81 | 82 | 81 |
| `libero_spatial/8` | 84 | 85 | 84 | 88 | 94 | 90 |
| **Successes / tasks** | **7/8** | **7/8** | **7/8** | **7/8** | **6/8** | **6/8** |

Both arms exhausted 300 controls on the drawer-and-bowl task (`libero_goal/3`) with every method. On the moka pots (`libero_10/8`), raw UR5e finished in 470 controls. Both corrections exhausted 520. The table uses the simulator's binary reward and termination for each episode. [Native reward readback](evidence/production-native.json)

### Raw success across layouts

| Arm | Failed both | Failed 13, solved 14 | Solved 13, failed 14 | Solved both | Raw successes on 13 → 14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IIWA | 1 | 1 | 0 | 6 | 6/8 → 7/8 |
| UR5e | 0 | 0 | 1 | 7 | 8/8 → 7/8 |

On IIWA the historical moka-pots failure already succeeded with raw actions on layout 14. On UR5e the historical drawer success failed on layout 14. This left exactly one eligible raw failure on each arm, the drawer. [Historical and production outcomes](evidence/production-native.json)

## Paired gains, losses and precision

The reference method is first. A gain solves a reference failure. A loss fails a reference success. The final column gives the candidate’s success rate minus the reference’s, in percentage points, with conservative paired 95% bounds.

| Arm | Reference → candidate | Both failed | Gained | Lost | Both solved | Rescue / eligible | Harm / eligible | Net pp [95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IIWA | Raw → planar | 1 | 0 | 0 | 7 | 0/1 | 0/7 | 0 [−42.2, 42.2] |
| IIWA | Raw → 3D | 1 | 0 | 0 | 7 | 0/1 | 0/7 | 0 [−42.2, 42.2] |
| IIWA | Planar → 3D | 1 | 0 | 0 | 7 | 0/1 | 0/7 | 0 [−42.2, 42.2] |
| UR5e | Raw → planar | 1 | 0 | 1 | 6 | 0/1 | 1/7 | −12.5 [−57.5, 42.0] |
| UR5e | Raw → 3D | 1 | 0 | 1 | 6 | 0/1 | 1/7 | −12.5 [−57.5, 42.0] |
| UR5e | Planar → 3D | 2 | 0 | 0 | 6 | 0/2 | 0/6 | 0 [−42.2, 42.2] |

One discordant task moves an arm’s net estimate by 12.5 points. For either correction against raw, 0/1 rescues has an exact 95% interval of 0–97.5%. The harm intervals are 0–41.0% for IIWA (0/7) and 0.4–57.9% for UR5e (1/7). For 3D against planar on UR5e, the intervals are 0–84.2% for rescue (0/2) and 0–45.9% for harm (0/6). The fixed panel establishes the observed ties and the single lost task. It has limited precision for effects across other tasks, layouts or noise draws. [Exact interval values](evidence/production-native.json)

The independent calculation used beta quantiles for the two-sided Clopper–Pearson intervals. The net bounds subtract simultaneous 97.5% exact bounds for the gain and loss probabilities. Interpreting these bounds beyond the panel assumes independent, exchangeable task identities. Both arms reuse the same eight identities. Each identity has one layout and one noise draw per arm. [Registered statistical method](report.py)

## Production validation

- **First episode.** Both first raw receipts were written at 20:25:42 CST. The 20:26 readback verified `libero_10/5` for all three methods on each arm. The native renderer reported `NVIDIA GeForce RTX 4090/PCIe/SSE2`, vendor `NVIDIA Corporation`. The two GPU probes attested distinct single RTX 4090 allocations. The pinned policy path requires one JAX GPU and calls frozen `pi05_libero` inference with explicit flow noise. [First check, GPU identities and runtime](evidence/production-native.json)
- **Pairing.** Across all 16 arm/task combinations, each method matched raw on the compiled arm model, full physical reset, unprocessed public observations, processed agent/wrist images, policy state, language instruction and flow-noise digest. I regenerated every noise bank from its query seed. Across arms, the released scene, row, task definition, perturbation plan, instruction, noise, action schema and horizon matched on all eight identities. Settled scene hashes differed across robot models. The largest recorded object-pose errors relative to the settled donor were 7.33 × 10⁻¹⁵ m and 2.68 × 10⁻¹⁴ rad. [Pairing evidence](evidence/production-native.json)
- **Opening proposals.** For every arm/task, all three receipts matched on the hash of the complete 10 × 7 pre-correction proposal. The runner retains that hash and the numerical five-control issued prefix. I recomputed planar and 3D XYZ for all 32 adapted prefixes using the registered matrices and clipping. Maximum absolute XYZ discrepancy was 5.96 × 10⁻⁸. Rotation and gripper components matched exactly. The frozen policy replans from its own evolving observations thereafter. All support tapes matched their hashes and used 15 controls on separate layout 11. [Proposal hashes and first-task commands](evidence/production-native.json)
- **Terminal outcomes.** I recounted the binary per-control rewards, controls and terminal flags from the 48 episode files. Every success ended with one unit of reward on its final control. Every failure exhausted its native horizon with zero return. Task outcomes, layout transitions, directional discordances and intervals agreed with both production summaries. [Receipt and summary hashes](evidence/production-native.json)

## Jobs, pins and GPU hours

Baige Distributed Training marked both jobs **Succeeded** on the `GPU-4090` cluster `aihc-sw5hua6i3hyv`, queue `aihcq-chqrelndaaj4`. Each used one card. All times below are November 23, 2026, CST (UTC+8). Scheduler GPU-hours use Baige’s reported task duration (任务耗时) times one card. Launcher GPU-hours use the measured job-step interval. [Timing and monitor record](evidence/production-native.json)

| Arm / job | Submitted | Launcher start → exit | Launcher seconds / GPU-h | Scheduler end | Scheduler duration / GPU-h |
| --- | ---: | --- | ---: | ---: | ---: |
| IIWA `job-uiifi561jv0n` | 18:44:25 | 20:13:19 → 20:29:31 | 972 / 0.2700 | 20:29:52 | 16m 31s / 0.2753 |
| UR5e `job-mwws9ocukkjq` | 18:44:37 | 20:13:21 → 20:29:32 | 971 / 0.2697 | 20:29:54 | 16m 31s / 0.2753 |
| **Combined** | | | **1,943 / 0.5397** | | **1,982 s / 0.5506** |

The queue checks ran from 18:46 through 20:07. Both launchers exited zero before the next 20-minute check following the first episode. The terminal scheduler readback was at 20:30 CST. Full receipts remain on `baige-dev` in `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-multiarm-confirmation-20261123-r1/gpu01-IIWA/` and its sibling `gpu01-UR5e/`. [Compact production evidence](evidence/production-native.json)

| Component | Verified revision or checkpoint |
| --- | --- |
| Executable | `cb246fffcec1d620603a7da841d29199568fcfa0` |
| Native policy/simulator runner | `a8a997e7d06bb131d3f9710204c58e106f0f0904` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Frozen checkpoint | `pi05-libero-openpi-20261027-r1` |
| LIBERO-Pro2.0 | `e5500c690c1498eca1b7050350654964190cdf70` |
