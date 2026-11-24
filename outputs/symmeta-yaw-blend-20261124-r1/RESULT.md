# Does blending SymMeta and tuned Perceiver yaw improve the frozen correction?

Across 768 known out-of-distribution (OOD) pairs, replacing the tuned yaw with an untuned 50:50 circular midpoint lowered mean absolute yaw error from **6.433° to 6.093°**. The paired change was **−0.340°** (95% whole-task interval **−1.049° to +0.447°**). Mean planar operator error fell from **0.1660 to 0.1612**. Both errors decreased on three of four panels and increased on UR5e π₀. Both pooled intervals allow a change in either direction. Keep the tuned reader as the baseline. This exploratory result on known cases gives insufficient evidence to switch, and native success for the blend remains unmeasured. [Computed results](metrics.json)

## Offline comparison

For each support and matched reader seed, the baseline uses the tuned Perceiver’s yaw, gain and gripper sign. The candidate moves halfway along the shorter arc from tuned yaw to the original SymMeta yaw and uses the same tuned gain and sign. The weight was fixed with no fitting. Yaw error is the absolute shortest-arc distance to the labeled yaw, in degrees. For a true planar map $A=gR(\theta)$ and an estimated map $\hat A=\hat g R(\hat\theta)$, operator error is $\lVert A\hat A^{-1}-I\rVert_F/\sqrt{2}$. Zero means an exact correction. Both columns of differences are blend minus tuned. [Label definitions](../../src/fluffy_bassoon/xemb/embodiment.py) · [Operator definition](../../src/fluffy_bassoon/xemb/wider_family.py)

| Arm and policy | OOD pairs | Yaw MAE, tuned → blend (°) | Yaw change [95% interval] (°) | Mean operator error, tuned → blend | Operator change [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| UR5e π₀.₅ | 192 | 5.822 → 5.083 | −0.739 [−1.725, −0.044] | 0.1583 → 0.1485 | −0.0099 [−0.0242, −0.0004] |
| IIWA π₀.₅ | 192 | 4.499 → 3.987 | −0.513 [−1.431, +0.488] | 0.1316 → 0.1232 | −0.0084 [−0.0232, +0.0071] |
| Kinova3 π₀.₅ | 192 | 5.832 → 5.592 | −0.241 [−0.809, +0.341] | 0.1590 → 0.1566 | −0.0023 [−0.0092, +0.0053] |
| UR5e π₀ | 192 | 9.578 → 9.710 | +0.132 [−1.150, +2.176] | 0.2151 → 0.2163 | +0.0013 [−0.0174, +0.0310] |
| **Pooled** | **768** | **6.433 → 6.093** | **−0.340 [−1.049, +0.447]** | **0.1660 → 0.1612** | **−0.0048 [−0.0157, +0.0069]** |

Each panel contributed eight shared tasks × eight OOD shifts (four compositional, four continuous-angle) × three matched seeds (41–43). The two readers’ three seeds are repeated measurements of the same tasks. Each of 20,000 bootstrap draws used Python’s `random.Random(1796)` to sample eight whole task identities (suite, task ID) with replacement. The same draw retained both predictions, every shift, every seed and all four panels. The 2.5th and 97.5th percentiles of the paired mean differences use linear interpolation. The intervals describe variation across these eight tasks, conditional on the seeds and panels. [Task identities and full-precision estimates](metrics.json)

Tuned and SymMeta gripper signs agreed on **1,152/1,152** paired predictions including familiar shifts. Both readers matched the labeled sign on **768/768 OOD pairs** (192/192 on each panel). The blend inherits that accuracy from tuned.

## Archived native outcomes for the original readers

On the 96 UR5e π₀.₅ compositional OOD pairs, **17 succeeded only with SymMeta and 4 only with tuned**. Both succeeded on 60 and both failed on 15. Among the 17 SymMeta-only successes, SymMeta had the smaller absolute circular yaw error on **15/17** and the smaller absolute gain error on **12/17**. Each original reader supplied its full correction. These outcome-selected counts leave the separate effects of yaw and gain unresolved. The CSV columns used here are `symmeta` and `tuned` from the paired native run. This audit used zero GPUs and collected zero new rollouts. [Paired native outcomes](../symmeta-perceiver-retry-20261123-r1/evidence/native02g4090/pairs.csv.gz) · [Original native study](../symmeta-perceiver-retry-20261123-r1/RESULT.md)

## Proposed test

Can the same 15 passive controls reveal which yaw estimate to trust on a particular trace? One candidate infers two yaw estimates from two of the three command chunks. For each estimate, an independent source-trained motion predictor scores how well the issued command explains the visual motion in the omitted chunk. Repeat for each omitted chunk, then use those scores to weight the full-trace yaw estimates. Fix the tuned gain and gripper predictions to isolate yaw. Prediction-error weighting has precedents in motor control. The empirical question is whether three passive chunks provide a transferable reliability signal. [Wolpert and Kawato (1998)](https://pubmed.ncbi.nlm.nih.gov/12662752/)

The first check would use unseen source tasks to determine whether these scores predict the better yaw estimate beyond constant weighting. A subsequent test would freeze the design and score fresh native retries against the tuned Perceiver, the fixed blend and a compute-matched Perceiver ensemble, using the same support, training data and paired resets. A gain confined to the inspected task subset or to offline yaw would leave the methods claim unsupported.

## Receipt and reproduction

The main-clone revision was `ecba79243dd5f7b0c360da50d759853d7306ed49`. The frozen prediction export used training revision `ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9`. Native evaluation used `d568535b758f2b97b382f9b3e280376e769b5942`. The analysis verified the following bytes on Baige, joined the two archives one-to-one by panel, case index and seed, checked suite, task ID, shift, regime and tape digest, and checked the full panel × task × seed × evaluation-shift grid.

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| [Frozen predictions](../symmeta-perceiver-retry-20261123-r1/evidence/train02pro/predictions.json) | 638266 | `c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b` |
| [Paired native outcomes, compressed](../symmeta-perceiver-retry-20261123-r1/evidence/native02g4090/pairs.csv.gz) | 98999 | `be37297b1950f4ec3463a55ce9e60617eba2318c42c363504d42844eb3d8cf1d` |
| [Evaluation labels and yaw-error function](../../src/fluffy_bassoon/xemb/embodiment.py) | 8170 | `e0ed86b5fc85bbde7b4dd95b0ec2b520b792325d0e6f135465132cc2dc99c41c` |

From the repository root, run the following with Python and NumPy (used by the label module). The script writes JSON to standard output.

```sh
python3 -B outputs/symmeta-yaw-blend-20261124-r1/analyze.py
```

The saved JSON was reproduced byte-for-byte with this read-only Baige Linux CPU command, run from the local repository root. The historical interpreter supplied Python 3.11.17 and NumPy 1.26.4. [Native runtime record](../symmeta-perceiver-retry-20261123-r1/LIVE.md)

```sh
run=outputs/symmeta-yaw-blend-20261124-r1
ssh -o BatchMode=yes baige-dev '
  cd /mowenjie/fluffy-bassoon-2 || exit 1
  src=setups/symmeta-perceiver-followups-20261123-r1/src-d568535b758f
  archive=outputs/symmeta-perceiver-retry-20261123-r1
  CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B - \
    "$archive/train02pro/predictions.json" "$archive/native02g4090-independent/pairs.csv.gz" \
    "$src/src/fluffy_bassoon/xemb/embodiment.py"
' < "$run/analyze.py" | cmp - "$run/metrics.json"
```

The uncommitted [analysis script](analyze.py) has SHA-256 `363b7dcff7515e17d511d282dde423e1d5a269624fbb7744263580cdb548286e`. The [computed JSON](metrics.json) has SHA-256 `7b5732d8c43536fb1a4c79c210b25ff6b7d53d6fa463922efceb24a04f7e106f`.
