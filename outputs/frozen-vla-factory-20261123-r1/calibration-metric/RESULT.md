# Lower yaw error identified 12 of 15 held-out native wins

Can offline calibration error rank two architectures by native success on individual cases? We joined the [matched-flow support estimates and native outcomes](../../symmeta-matched-flow-20261123-r1/RESULT.md) at revision `757a44cd4374833cad790852df8b87ddf412e638` by index and checked task, interface and regime. Each original pair shared its UR5e reset, frozen π₀.₅ proposal and noise schedule. Gain and gripper corrections were fixed across the two readers. The support comprised 15 controls.

| Historical native assignments | Always learned | Always linear | Choose smaller true yaw error | Either reader succeeds | Successful choice on discordant pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 96 | 70 | 62 | 73 | 76 | 17/20 |
| Held-out 64 | 46 | 39 | 47 | 50 | 12/15 |

The lower-error choice gained 1.56 percentage points over always using the learned reader on held-out assignments (95% paired eight-task bootstrap interval −4.69 to +9.38). Its three misses on discordant pairs are counterexamples to ranking these retries solely by yaw error. This is a retrospective diagnostic using privileged ground truth to choose a recorded branch. The experiment supplies no deployable selector or new rollouts.

Reproduce from the repository root with `uv run --no-sync python outputs/frozen-vla-factory-20261123-r1/calibration-metric/judge_yaw.py`. The bootstrap used 20,000 draws with seed 20261123. [Machine-readable counts and intervals](result.json)
