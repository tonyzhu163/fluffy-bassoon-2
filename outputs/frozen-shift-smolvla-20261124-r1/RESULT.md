# Result: Kinova3 0 of 12, Panda 3 of 12, IIWA 2 of 6; direction matches, sample too small to settle

[Plan](PLAN.md) · [Usage](README.md) · [`summary.json`](summary.json). Code at branch `devin/1795493350-frozen-shift-smolvla`. Not a confirmatory result.

Every success below is fresh **native reward** on CPU: frozen SmolVLA (`a24f1f38`, 450,046,176 parameters, readback unchanged in all 40 episodes), LIBERO-Pro2.0 `e5500c69` BASE `libero_goal/3`, robosuite 1.4.0, MuJoCo 3.2.3, Mesa EGL, 300-control horizon, 50 controls per proposal, official LeRobot 0.4.4 preprocessing.

## Instruction wording decides whether Panda works at all

| Instruction | Panda | Kinova3 |
| --- | --- | --- |
| Pro2.0 BASE: "open the top layer of the drawer and put the bowl inside" | 0/7 (14, 15, 16, 17, 36, 37, 38) | 0/1 (14) |
| LIBERO release: "open the top drawer and put the bowl inside" | 2/6 same noise | 0/6 |

Layouts 14 and 36 flip from fail to success on Panda with only the wording changed (same reset, same noise). All later rows use the LIBERO wording through the `prompt` hook ([plan amendment 1](PLAN.md)).

## Matched starts, LIBERO wording

Cell: success or fail, controls consumed. Same reset seed and same noise sequence across a row.

| Layout | Panda, noise 1 | Kinova3, noise 1 | IIWA, noise 1 | Panda, noise 2 | Kinova3, noise 2 |
| ---: | --- | --- | --- | --- | --- |
| 14 | success 183 | fail 300 | success 180 | fail 300 | fail 300 |
| 15 | fail 300 | fail 300 | fail 300 | fail 300 | fail 300 |
| 17 | fail 300 | fail 300 | fail 300 | success 264 | fail 300 |
| 36 | success 175 | fail 300 | fail 300 | fail 300 | fail 300 |
| 37 | fail 300 | fail 300 | success 176 | fail 300 | fail 300 |
| 38 | fail 300 | fail 300 | fail 300 | fail 300 | fail 300 |
| total | 2/6 | 0/6 | 2/6 | 1/6 | 0/6 |

- Panda 3/12 (exact 95% interval 5% to 57%), Kinova3 0/12 (0% to 26%), IIWA 2/6 (4% to 78%).
- Panda against Kinova3: Fisher exact two-sided p = 0.22 pooled; paired sign counts 3 Panda-only, 0 Kinova3-only. Panda and IIWA pooled (5/18) against Kinova3 (0/12): p = 0.066, a comparison chosen after seeing the data.
- Untreated repeat with the same noise, layout 14: Panda success 183 again, Kinova3 fail again, command digests identical on both arms. The harness is deterministic, so success varies only through the noise seed.
- Fresh noise changes the Panda outcome on 3 of 6 starts (14, 17, 36). No layout is solved by Panda under both noise seeds.
- Scene fingerprints match across arms on every row; target-arm reset IK position error is at most 0.04 mm.

## Reading

- The target-arm direction is the same as π₀.₅: Kinova3 never earned reward, Panda and IIWA did, IIWA on layouts 14 and 37, the two layouts where a π₀.₅ IIWA command tape replayed on Kinova3 earned reward. It is not established: 12 episodes per arm cannot separate 0% from Panda's 25%.
- SmolVLA is a weak motor on these six starts (source-arm success about 28%), so one episode on this bed carries little information. An adapter test here needs several noise seeds per start and a matched untreated control.
- Wording and policy noise each change the Panda outcome with weights untouched. Both are levers the proposal-shift program already names; proposal choice over noise seeds has measured headroom on Panda (3 starts solved by one of two seeds).

## Evidence kinds

| Kind | What |
| --- | --- |
| Native reward | 40 episodes in `evidence/` (`raw` 8, `libero-prompt` 18, `libero-prompt-noise2` 12, `libero-prompt-same` 2), logs in `logs/` |
| CPU fixture | `test_harness.py`, 5 tests on a toy arm and scripted motor: determinism, timeout, clipping, hooks, noise shape. No simulator, no weights. Not task evidence |
| Offline forward-pass | none new |
| Archived replay | none |

## Not done

- Layouts 16, 31, 34, 35, 39 (source-arm development successes) with the LIBERO wording, UR5e, other `libero_goal` tasks, more noise seeds.
- No adapter was evaluated beyond the instruction wording. `Replan10` hooks exist; two probes were stopped unfinished and are not evidence.
- `setup.sh` was written from the commands that built the scratch used here but was not rerun from an empty directory. `run.sh` and `harness.py` are the exact code behind every episode.
- Kinova3 failure stage (drawer opened or not) was not audited.
