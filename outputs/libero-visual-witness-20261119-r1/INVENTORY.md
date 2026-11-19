# Frozen π0.5 two-destination tape inventory

**Use the bowl tapes for visual replay.** Both destinations were physically executed on each of eight development resets, 12–19, outside the specified matching and interrupted-bottle ranges. The released D18 bottle pairs have zero eligible resets and no saved camera frames. “Fresh” here means outside the specified ranges. The bowl resets were used in earlier development work.

All artifact paths below are relative to **Baige** `/mowenjie/fluffy-bassoon-2/outputs/`.

## Released D18 bottle pairs

Under `libero-interrupted-practice-20261119-r1/`, the saved branches are:

| Physical branch | Episode path | Actual coverage |
|---|---|---|
| Released D18 drop | `release02/episodes/p{p}-g{g}-D18/` | 12 tapes, resets 24–29 × goals 0–1. All 12 result records contain a release event. The earlier `release01/episodes/` has zero completed episodes. |
| Matched sham | `closed01/episodes/p{p}-g{g}-sham/` | All 12 corresponding reset/goal pairs |
| Independent query | `closed01/episodes/p{p}-g{g}-query/` | Both query options on each of the same six resets, 12 tapes |

Goal 0 is wine bottle to drawer top (LIBERO-Goal task 2), goal 1 is wine bottle to rack (task 9). Each episode has `result.json` and `tape.npz`. The five tape arrays are `issued`, `states`, `hits`, `contact` and `heights`. CPU replay can score saved native hits and read robot motion. A camera-based drop witness would need images from another source. The open-loop `drop01/drop.json` is a readout of the same 12 pairs. The collector describes matching the sham prefix and using a separate noise seed for query. [Collector](../libero-interrupted-practice-20261119-r1/live.py) · [Released scorer](../libero-interrupted-practice-20261119-r1/released_score.py)

## Bowl tapes with both camera views

Under `libero-bowl-decisions-20261118-r1/`, goal 0 is bowl to drawer top (LIBERO-Goal task 4), goal 1 is bowl to plate (task 8). Both prompts run in the task-4 world. Each row below covers both goals and both variants, BASE and S20. All 64 episode records have 300 controls, and the starting physics and public-observation fingerprints match across options and modes within each reset/variant.

| Episode path | Reset IDs | Physical branches |
|---|---|---|
| `paired01/episodes/s{p}-{variant}-g{g}-{mode}/` | 12, 13, 14, 15 | `gain1` and `gain0.5`, 32 episodes |
| `pacing01/episodes/s{p}-{variant}-g{g}-{mode}/` | 16, 17, 18, 19 | `gain1` and `paced`, 32 episodes |

Both `episodes/motor.json` receipts identify checkpoint `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1` and OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479`. Each `complete.json` records 32 episodes. [Collector and camera order](../libero-bowl-decisions-20261118-r1/pilot.py) · [Goal mapping and native scoring](../libero-bowl-decisions-20261118-r1/sim.py)

**Feasible CPU replay.** Use BASE `gain1` on 16–19 as support and BASE `gain1` on 12–15 as queries. These roles are assigned offline to disjoint physical resets. Each side supplies eight real branches at the same five-control cadence. In support, drawer succeeds on 16–18 and fails on 19, and plate succeeds on all four. In queries, each prompted destination succeeds on all four resets and the opposite destination never registers. This supports scoring a destination choice and gives only one failed nominal support example. S20 `gain1` has zero intended hits on all eight resets for both prompts. The alternative `paced` branches succeed on all eight S20 reset/goal combinations on 16–19. The bowl manipulations change command gain or cadence and cannot establish the effect of a D18 drop.

Read `public.npz` with NumPy. The nominal `frames` shape is (60, 2, 224, 224, 3), ordered agent view then wrist view. `final_images` supplies both terminal views, `issued` is (300, 7), and `state_after` is (300, 9). In `paired01`, nominal frame indices map to controls 0, 5, …, 295. `pacing01` saves `decision_steps` explicitly. Keep `evaluator-predicates.npy` separate from visual inputs. For chosen option g and assigned destination h, read that option's physical branch and score `hits[:, h].any()` for ever success or `hits[-1, h]` for final success. The saved native array is (300, 2). This replay requires neither inference nor simulation.

## Bottle alternative

`libero-first-attempt-20261119-r1/transfer02/episodes/bottle/s{p}-{variant}/support-g{g}-*/` has 48 RGB-bearing physical supports on resets 0–5, covering both bottle goals, both variants and both post-prefix cadence choices. Use `result.json` field `paced` to identify the executed cadence because one directory per block is labeled `learned`. The 24 sibling `query-g{g}/` branches use physical reset **p + 6**, namely 6–11, all excluded. The parent directory encodes the support reset. This source supplies eligible support tapes but no eligible originally designated queries. [Transfer collector](../libero-first-attempt-20261119-r1/experiment.py)

Inventory checks used read-only Baige episode records, tape member names and array headers, plus the project collectors. Outcome counts use saved native first-hit records. No GPU or simulator was used.
