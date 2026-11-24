# Result: the rotation sign branch decides frozen SmolVLA's outcome on both arms

[Plan](PLAN.md) · [`summary.json`](summary.json) · adapter [`adapters_state.py`](adapters_state.py). Episodes ran at revision `2f2085f3c231fe89da05228af62d899432f8714d`. Exploratory, 48 episodes.

Every outcome is fresh **native reward** on CPU: frozen SmolVLA `a24f1f38` (weight readback unchanged in all 48 episodes), LIBERO-Pro2.0 `e5500c69` BASE `libero_goal/3`, layouts 14, 15, 17, 36, 37, 38, LIBERO-release wording, two policy-noise seeds. It is not evidence about π₀.₅.

| Arm | State rotation X over the episodes | Successes | Exact 95% interval | Solved (layout, noise seed) |
| --- | --- | ---: | --- | --- |
| Kinova3 raw | −3.57 to −2.81 | 0/12 | 0% to 26% | none |
| Kinova3 mapped to the source branch | +2.84 to +3.15 | 4/12 | 10% to 65% | 14 and 37, both seeds |
| Panda raw | +2.93 to +3.20 | 3/12 | 5% to 57% | 14 and 36 (seed 1), 17 (seed 2) |
| Panda flipped to the Kinova3 branch | −3.37 to −2.85 | 0/12 | 0% to 26% | none |

- Mapping rescued 4 Kinova3 episodes and lost 0. Flipping lost all 3 Panda successes and gained 0. All 7 changed outcomes go the predicted way (two-sided sign test p = 0.016 over both directions; 0.125 and 0.25 within each arm).
- Pooled by branch: 7/24 on the source branch, 0/24 on the Kinova3 branch.
- The adapter changes one representation of one channel. The rotation is identical (unit-tested), and images, prompt, noise, position and fingers are untouched. Within each start and noise seed all four arms drew the same policy noise, and both arms of a pair share the scene fingerprint.
- Kinova3 raw reproduced the earlier run's command digests on this second machine, so the 0/12 is the same deterministic baseline.
- Mapped Kinova3 solves layouts 14 and 37, the two that IIWA solved with SmolVLA and the two where an IIWA π₀.₅ command tape replayed on Kinova3 earned reward.

## Reading

- For this motor the Kinova3 failure on these starts is an observation-interface failure. The simulator reports the Kinova3 hand quaternion with the opposite sign, the axis-angle conversion turns that into a 2π shift, and the policy has never seen that value. With the branch fixed, Kinova3 (4/12) is level with Panda (3/12) on a panel where 12 episodes cannot separate them.
- It is not established that the branch explains the π₀.₅ Kinova3 failures. π₀.₅ solves other Kinova3 layouts and tasks with the raw sign, so any effect there is partial. Every native π₀.₅ Kinova3 run in this repository packed the raw sign.
- Limits: one task, six layouts chosen because π₀.₅ failed on them, two noise seeds, a motor that solves about a quarter of source starts. IIWA and UR5e were not rerun here.

## Next

Run `state-mapped` and its controls on π₀.₅ through the [program launcher](../frozen-vla-proposal-shift-20261124-r1/run.sh) before any other lane. If Kinova3 recovers, rerun the earlier Kinova3 comparisons on the mapped state.

## Reproduce

```bash
bash ../frozen-shift-smolvla-20261124-r1/setup.sh $SCRATCH
bash run.sh $SCRATCH kinova3-mapped-noise0 Kinova3 SourceBranch 0 14 15 17 36 37 38   # also Raw, and Panda Raw / TargetBranch, offsets 0 and 1000003
python summarize.py
uv run --no-sync pytest -q -W error outputs/frozen-shift-smolvla-state-20261124-r1
```
