# Native probe-order collection

The planned panel has **128 blocked supports, 128 shuffled supports and 128 independent queries**, each with 16 measured actions. The experimental unit is a layout index shared across two simulated arms and four yaw–gain settings. A complete panel contains 384 tapes and 6,144 measured actions. Design base `4bd494d188bfdcb757c554bb335b5c19d3bc97e6`. [Frozen design](../embodiment-tape-observability-20261120-r1/RESULT.md) · [Run plan](PLAN.md) · [Collector](collect.py)

| Tape at index `i` (0–15) | Donor seed | Public command sequence in time order |
| --- | --- | --- |
| Blocked support | `991200+i` | Hold +X, −X, +Y, −Y for four actions each. Each translation has amplitude 0.39, the other motion components are zero, and the jaw command is −1. |
| Shuffled support | Same support donor | Apply NumPy `default_rng(2991200+i).permutation(16)` to the blocked rows. The receipt saves the zero-based source row for each issued step. |
| Shared query | `991300+i` | Call the pinned H1 mixed generator with seed `2991300+i`. It draws 16 six-component commands uniformly from [−0.4, 0.4], then 16 jaw commands from {−1, +1}. |

For each index the collector constructs the three public arrays once and reuses them across Panda, Sawyer and all four combinations of yaw +45°/+90° and gain 1.0/0.65. Both supports in a cell use the **same single query row** for scoring. The row key is `(family, index, yaw_deg, gain, schedule)`. Replace only the support yaw or only its gain for a one-factor swap. Keep the index, family, support schedule and receiving query fixed. [Frozen scorer](score.py)

## What the controls match

Each support has eight X-axis and eight Y-axis actions, zero net commanded translation and a duration of 0.8 s at 20 Hz. Each has nominal public translation energy `16 × 0.39² = 2.4336`, measured as the sum of squared XYZ command components. The private interface issues `gain × Rz(yaw) × public_xyz`, with positive yaw rotating +X toward +Y. Angular and jaw commands pass through. At gain 0.65 the nominal issued energy is 1.028196. The files record energies from the actual float32 command values. Before either support executes, the collector checks exact row equality under the same permutation on the mapped controls. Native readback checks every issued action. Equal command energy leaves realized wrist motion free to differ.

The native sampler draws 32 BASE Goal 5 Panda donors. R2 checks the published 50-row bank, task and seed, contact admission, initial success and the fixture/joint scene. The collector also requires unique hashes across the 32 donors for the raw state, scene, movable-object XY and free-joint XY. Each tape starts from a fresh R2 reset. Within a family and donor, all eight support openings or all four query openings must have identical settled state and scene fingerprints and identical eight-component public opening observations. Across arms the common controls are the donor scene, seed and public commands. Sawyer uses bounded IK to transfer the scene and initial grip pose. Its reset receipt records the resulting cross-arm differences in settled position, rotation and jaw opening. [Native placement and reset](../libero-paligemma-dualarm-20261120-r2/generated_layouts.py) · [Arm reset and native step](../libero-crossarm-20261120-r1/arm_env.py)

## Time and native receipts

Indices run from 0 through 15. For each index, both arms finish their supports before either arm begins its query block. Each block visits Panda then Sawyer, with cells ordered (+45°, 1.0), (+45°, 0.65), (+90°, 1.0), (+90°, 0.65). Within a support cell, blocked runs first when the sum of the zero-based index, arm position and cell position is even. Shuffled runs first otherwise. Each schedule runs first at eight of the sixteen indices within every arm/setting. This alternation only controls acquisition order. The time order inside each tape follows its public command array.

| Raw JSONL fields | Meaning |
| --- | --- |
| `family`, `index`, `yaw_deg`, `gain`, `schedule`, `role`, `layout_seed` | Analysis and pairing keys. Role is `support` for both practice schedules and `query` for the shared query. |
| `commands`, `motion` | Public 16 × 7 pre-map commands and 16 × 3 measured world-frame wrist displacements in metres. These are the reader arrays. Row zero pairs the first command with its subsequent displacement. |
| `issued_actions`, `worker_step_index`, `wrist_world_m` | Audit-only native action readbacks, native action indices 1–16, and 17 world wrist observations beginning at reset. For every step the collector checks displacement against successive observations within the H1 absolute tolerance of 1e−7 m. |
| `rewards`, `success`, `terminated`, `truncated` | Four time-ordered arrays from the 16 native step packets. An episode ending before packet 16 stops collection. |
| `reset`, `opening_state`, placement hashes, `layout_id` | R2 model and environment receipts, contact checks, alignment and donor identity. The sampler and executing arm are checked for disabled on-screen rendering, offscreen rendering and camera observations. |
| `tape_ordinal`, `started_utc`, `finished_utc`, `probe_seed`, `blocked_row_indices` | Actual one-based acquisition order and UTC tape boundaries. The fixed blocked tape has a null probe seed. Only shuffled rows carry the permutation. |
| `translation_energy`, `issued_translation_energy` | Public and mapped sums of squared XYZ controls. Both omit angular and jaw components. |

The raw artifact includes private condition labels and issued-action receipts for auditing. Reader inference uses the public command and motion channels. A motion label comes from the observation returned **after** the action in the same row. Native reset settling precedes the first row and is excluded from the 6,144-action budget. [H1 mixed generator and transition convention](../embodiment-command-frame-20261120-r1/collect.py) · [Camera-free R2 session](../libero-paligemma-dualarm-20261120-r2/expert.py)

## Publication and partials

The Linux launcher invokes `collect.py --out "$out/collection"` in a fresh directory using the qualified R2 interpreter. The collector flushes completed rows to `tapes.jsonl.partial` and updates `collection.json` after each role/index block. The manifest includes the checkout revision, source digests, donor states and receipts, tape count, native calls started, and native packets returned. It publishes `tapes.jsonl` only after checking the full set of 384 unique keys, 32 donor identities and both native counts of 6,144. [Launcher](run.sh)

A reset mismatch, rejected donor, early terminal, bad native packet or other handled exception stops the attempt with a nonzero exit. The manifest records the error and last active block or tape. When a tape was in progress, `partial-tape.json` preserves its planned public commands and the verified native prefix available at failure. A started step can raise after physics advances, so only a returned packet increments the return count. SIGINT and SIGTERM request a stop at the next collection boundary. A hard kill may leave a running manifest at its last checkpoint. Use a fresh directory for a diagnosed retry.

**Implementation verification.** macOS syntax compilation and Ruff checks passed. Static inspection enumerated the 32 donor seeds and all 384 keys, checked the axis order, multiplicities, shuffle and H1 query calls, and matched the rotation and gain expressions to H1. The support energies above follow from the specified controls. This implementation check produced zero native tapes and zero simulator actions. Native reset equality, measured motion and success require receipts from Linux execution.
