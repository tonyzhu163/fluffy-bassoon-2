# Instruction-swap check: results

Run `pi05-instruction-swap-thu2-20261029-r1`, [plan](PLAN.md). Frozen pi0.5,
RLinf eval-mode sampling, no training. Secondary school cluster (thu2),
source `8baf75b48fe21d1eacaae7c10a032b55dbd7f72d` (branch
`instruction-swap-maniskill`). Raw outputs:
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/pi05-instruction-swap-thu2-20261029-r1/<ckpt>-job<id>/`;
small receipts are mirrored locally under the same subdirectories.

## Jobs

| Checkpoint | thu2 job | Node / GPU | Wall | Result |
| --- | --- | --- | --- | --- |
| RL `RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise` step 150 | **49180** | node02, Slurm GPU IDX 4 (`GPU-6e52b20d`), probe passed | 21:27 | exit 0; 57 tests passed in-step; 64 single + 288 multi episodes, all 3 blocks complete; frozen fingerprint `ed862a89...` unchanged (matches Baige) |
| SFT `RLinf-Pi05-ManiSkill-25Main-SFT` | **49194** | not started | - | **queued** (submitted 2026-10-29 04:28 UTC, same commit/launcher). At 04:40 UTC Slurm reported `ReqNodeNotAvail` with no start estimate: node02 had 2 free GPUs but 184/188 GiB RAM allocated (48G requested), node01/03/04 had no free GPU. It runs unattended if resources free up and writes `sft-job49194/metrics.json` itself |

CPU tests before launch: thu2 job 49179 (node01, CPU only), 59 passed.

## Metrics

Explicit success = named object on the plate (RLinf success test applied to
that object) at any step. Chosen object = placed, else first grasped.
Swap-follow = chosen object equals the named object under both prompts of a
paired scene. Gap = explicit success - vague success for the same object
over the same 24+24 scenes. Each object: 16 single-object scenes and 48
multi-object scenes per condition (24 per pair, 12 per slot order).

| RL checkpoint | Single-object success | Explicit success (2-object) | Explicit, only named placed | Other object placed when named | Vague success | Gap (pts) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| carrot | 94% (15/16) | 52% (25/48) | 52% (25/48) | 0% (0/48) | 35% (17/48) | +17 |
| toy bear | 94% (15/16) | 19% (9/48) | 19% (9/48) | 19% (9/48) | 21% (10/48) | -2 |
| banana | 100% (16/16) | 77% (37/48) | 71% (34/48) | 8% (4/48) | 46% (22/48) | +31 |
| hamburger | 81% (13/16) | 23% (11/48) | 19% (9/48) | 33% (16/48) | 12% (6/48) | +10 |

Swap pairs (RL): 96 paired scenes, 89 with any grasp.

| Swap metric (RL) | All pairs | Pairs with any grasp |
| --- | ---: | ---: |
| Chosen object changes with the prompt | 28% (27/96) | 30% (27/89) |
| Chosen object = named object under both prompts | 28% (27/96) | 30% (27/89) |
| Only the named object placed under both prompts | 9% (9/96) | |

Vague prompt (`put object on plate`), chosen object by pair (RL):

| Pair | Counts | Majority share among acted |
| --- | --- | ---: |
| carrot/toy bear | carrot 12, none 2, toy bear 10 | 55% |
| banana/hamburger | banana 14, hamburger 9, none 1 | 61% |
| carrot/hamburger | carrot 12, hamburger 6, none 6 | 67% |
| toy bear/banana | banana 13, none 1, toy bear 10 | 57% |
| all | 86/96 acted | 59%; slot-0 object 72% |

## Criteria (provisional, fixed before launch)

| Criterion | RL | SFT |
| --- | --- | --- |
| Explicit success >= ~60% per object | **fail**: banana 77%, carrot 52%, hamburger 23%, toy bear 19% | pending |
| Swap-follow >= ~80% | **fail**: 28% all pairs, 30% given a grasp | pending |
| Vague near chance among objects | object identity near chance (pooled majority share 59%), but slot-0 object chosen 72% (positional bias) | pending |
| Gap >= 30 points per object | **fail**: banana +31, carrot +17, hamburger +10, toy bear -2 | pending |

## Post-hoc diagnostics (not part of the pass criteria)

Computed from the RL `multi/episodes.jsonl` (chosen object by slot; 96 scenes each):

| Prompt | Chosen slot-0 object | Chosen slot-1 object | None | Slot 0 placed | Slot 1 placed |
| --- | ---: | ---: | ---: | ---: | ---: |
| names slot-0 object (RLinf native prompt) | 69 | 7 | 20 | 49 | 10 |
| names slot-1 object | 33 | 46 | 17 | 19 | 33 |
| `put object on plate` | 62 | 24 | 10 | 42 | 13 |

Explicit success by the named object's slot (RL): banana 20/24 at slot 0,
17/24 at slot 1; carrot 13/24, 12/24; hamburger 10/24, 1/24; toy bear 6/24,
3/24. RLinf's MultiCarrot layout places the slot-0 object closer to the
plate on average (0.126 m vs 0.170 m start distance; the slot-1 object only
has to be 0.15 m from the slot-0 object), so the slot-0 preference is partly
geometry, not language. All conditions of a scene used identical episode
ids, reset seed and sampler seed. Slot-0 on-plate outcomes matched RLinf's
own success flag at every step (0 mismatches). The frozen-weight
fingerprint did not change. Peak CUDA memory was 12.0 GiB at 32 envs, and
each three-prompt block of 32 scenes took about 300 s.

## Interpretation

RL checkpoint: **does not pass**. The language instruction does steer
object choice, but weakly and unreliably. Naming the slot-1 object raises
how often slot 1 is chosen from 7/96 to 46/96 scenes. Only 28% of paired
scenes follow both prompts, and for 3 of 4 objects explicit success in
two-object scenes (19-52%) falls far below single-object success (81-100%).
The vague prompt behaves like a nearest-object default (slot 0 chosen 72% of
the time). So the gap is small: only banana reaches +30 points.

Implication for the hidden-target ICRL task: with this motor, the
instruction cannot reliably choose the target. Explicit naming of the
correct object gives at best ~50% target success on average and is
object-dependent, so an "oracle-named" upper bound would sit far below
single-object competence. Reward-driven target identification would be
confounded by motor failures on the distractor scene. Any revised design
should treat explicit-name competence as the headroom ceiling. Options are a
different selection channel (for example position or an image cue), objects
with high multi-object explicit success (banana-like), or a motor fine-tuned
with distractors. The SFT replicate is pending, and these are 48 scenes per
object per condition: one seed, a single plate and four objects.
