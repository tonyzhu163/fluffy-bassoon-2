# Sampling lane result

[Plan](PLAN.md). Stopped at the 50-minute budget. **No native reward and no policy forward pass were produced.** Nothing below is task success. Evidence classes: archived native outcomes (section 1), offline fit on archived source trajectories (section 2), CPU fixture (section 3).

## 1. What archived outcomes bound about noise headroom (archived native outcomes)

`mine.py` found 47 Kinova3 drawer-task records in JSON under `outputs/` ([archived-episodes.json](archived-episodes.json)). Untreated ones with more than 20 controls:

| Layout | Untreated records | Successes | Distinct policy-noise digests | Source |
| --- | ---: | ---: | ---: | --- |
| 14 | 2 | 0 | 1 (`962c85c82c`) | visual-adapter raw, pose replay of the same tape |
| 15 | 2 | 0 | 1 (`a9d625a5ab`) | same two runs |
| 17 | 2 | 0 | 1 (`0b2166c19c`) | source-view raw and control |
| 36 | 1 | 0 | 1 | donor own-tape replay |
| 37 | 1 | 0 | 1 | donor own-tape replay |
| 38 | 0 | – | – | none parsed |
| 16 | 2 | 2 | 1 | source-view raw and control |
| 34, 35 | 1 each | 1 each | 1 each | donor own-tape replay |

- Every parsed failure on a start used the one noise stream of seed 41810300 + layout. The archive therefore holds **zero independent policy-noise draws** on the failed starts and puts no bound on per-start success frequency under fresh noise. One failure per start is compatible with any frequency below 1.
- The 2/6 to 4/6 spread across jobs ([factory](../frozen-vla-factory-20261123-r1/RESULT.md)) is variation under the same noise seeds, and those six starts span three tasks. I did not re-derive it. It shows outcomes move without a noise change, which is why the panel keeps `repeat`.
- What 0/8 would bound: an exact 95% upper limit of 0.37 on one start's frequency; 0/48 pooled gives 0.074.
- Not parsed (other record formats): selector launch, frozen-adaptation, donor-validation, cue, timing, identity, supported-correction, multiarm-confirmation, `libero-retry-headroom` (eight tasks, no Kinova3). The table is a lower count of archived repeats.
- Archived seeds are 41810300 + layout; the harness panel uses 42710300 + layout. If the seed also feeds the noise bank (it does in `ur5e_motor.run_episode`), the panel's `raw` is already a new noise stream relative to the archived failures. The owner of the harness should confirm which seed is intended.

## 2. Source critic (offline fit, archived source trajectories)

Source drawer-task trajectories with poses in the repo: six Panda π₀.₅ episodes on layouts 14 and 15 (3 successes, 3 failures, [pose treatment](../frozen-transfer-pose-20261124-r1/evidence/treatment)) and two human Panda demonstrations (`goal3.h5`). **No UR5e drawer trajectory with poses is archived**, so the critic is Panda-only. Fitted on the 5 successes; ridge on 14 quadratic features of displacement from reset and commanded gripper state; weights in [critic.json](critic.json).

Held out by episode, and by layout for the π₀.₅ episodes:

| Check | Result | Chance |
| --- | --- | --- |
| Later frame scored above earlier frame, ≥25 controls apart, π₀.₅ successes | 41,862 / 43,937 (95.3%) | 1/2 |
| Same, human demonstrations | 22,900 / 23,788 (96.3%) | 1/2 |
| Executed five-control window ranked first of four (decoys from ¼, ½, ¾ of the episode later) | 29 / 114 (25.4%) | 1/4 |
| Same by episode quarter | 0/30, 0/30, 7/28, 22/26 | 1/4 |
| Success scored above failure, final 20 controls | 5 / 9 pairs | 1/2 |

Reading: the critic orders frames in time but does not rank the executed window above alternatives (chance overall). The quarter split shows it prefers whatever window moves toward late-episode states, so it tracks decoy phase and not window quality. It does not separate Panda successes from failures. Three held-out π₀.₅ episodes on two layouts; no interval is meaningful. This counts against the critic before any native run. The decoys are other-time windows, not other noise samples at the same state; same-state proposals are not archived, so the native comparison against `random-k4` is still the deciding test.

## 3. Delivered code (CPU fixture)

| File | Content |
| --- | --- |
| [adapter.py](adapter.py) | `ADAPTERS`: `critic-k4`, `random-k4`, `single-k1`, `noise-0` … `noise-7` |
| [critic.py](critic.py) | Fit, held-out checks, proposal scoring with the 15-control calibration response |
| [headroom.py](headroom.py) | Per-start frequency and exact binomial intervals from panel output |
| [mine.py](mine.py) | Archive scan |
| [run.sh](run.sh) | Three calls of the harness launcher, then `headroom.py` |
| [test_sampling.py](test_sampling.py) | 7 checks |

`uv run --no-sync pytest -q -W error outputs/frozen-shift-sampling-20261124-r1` gives 7 passed; the harness suite still gives 25 passed with this lane discovered. On the fixture: only noise digests, candidate count and chosen index differ from the untreated loop; `critic-k4` and `random-k4` receive identical candidates and `single-k1` the first of them; `noise-k` streams differ across `k`, repeat for a start and differ across starts; the critic's choice is the argmax of the stored weights.

Launch on one Baige RTX 4090 (not run): `bash outputs/frozen-shift-sampling-20261124-r1/run.sh <pin> <attempt>`. Episodes: 6 × (2 + 8) + 5 × (2 + 2) + 11 × (2 + 3) = 135, about 0.5 GPU-hours at 0.0037 per episode, more for the K=4 arms' extra policy calls.

## Not done

- Any native episode. Headroom and the critic comparison are unmeasured.
- `run.sh` was not executed; only its three panel argument sets were resolved through `panel.discover` in a test.
- `noise-k` adapters are in `ADAPTERS`, so an unfiltered shared panel run would add eight branches per start. Name adapters with `--adapters`.
- An unfiltered full-archive count of repeats (formats above), and a critic with scene input. `h5py` is not in the repo environment; refitting needs it, running the adapters does not.
- arXiv 2610.22576 was cited from the repository's note only.
