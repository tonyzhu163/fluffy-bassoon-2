# One passive episode supports a scaled rotation on these three arms

On four shared held-out tasks, frozen π₀.₅'s commanded XY motion predicted the next episode’s recorded XY displacement with **6.58 mm** mean error per five-control chunk using a scaled rotation. Adding a rank-one target morphology correction to the Panda/Sawyer response prior gave **7.94 mm**. Its paired change versus the shared prior was **−0.02 mm** (95% whole-task interval **−0.13 to +0.04**). Its excess over the scaled rotation was **+1.36 mm** (**+0.50 to +2.22**).

Keep the scaled rotation as the passive-support baseline. The rank-one update has no consistent advantage here. For an unrestricted response map, test an independently directed calibration episode before spending compute on a native retry. The measured outcome here is cross-task motion prediction on simulated arms. Native task success under these adapters is unmeasured.

## Paired evidence

Each table entry is mean XY Euclidean endpoint-displacement error in mm per chunk, averaged over three chunks, two choices of calibration task, interfaces and tasks. Lower is better. The common row balances all three arms on the same four tasks and 12 persistent settings. The individual rows use every qualified held-task episode for that panel.

| Arm and frozen motor | Scored episodes | Scaled rotation | Shared Panda/Sawyer | Rank-one update | Full target nominal map | Unrestricted support map |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All three, π₀.₅, common tasks | 144 | **6.58** | 7.96 | 7.94 | 23.57 | 21.71 |
| UR5e, π₀.₅ | 72 | **5.88** | 8.49 | 8.29 | 30.04 | 21.58 |
| IIWA, π₀.₅ | 60 | **5.99** | 7.56 | 7.46 | 22.33 | 15.37 |
| Kinova3, π₀.₅ | 60 | 6.56 | **5.99** | 6.24 | 14.16 | 24.63 |
| UR5e, π₀ | 72 | **5.45** | 7.30 | 7.16 | 27.33 | 19.34 |

On the common panel the rank-one correction reduced error versus the full target nominal map by 15.64 mm (8.41 to 22.86). Giving the scaled rotation the next archived interface's support, cyclically, raised its error to 68.38 mm. On UR5e, the separate 288-control UR5e probe map plus the same unknown-interface support gave 6.10 mm on π₀.₅ and 5.60 mm on π₀.

Two follow-up controls clarify the result. On the common **nominal-only** subset, before fitting any interface rotation, target calibration reduced the shared template's error from 13.98 to 6.90 mm. Scalar gain adjustment alone gave 6.97 mm (rank-one minus scalar −0.06 mm, interval −0.16 to +0.08). Spending **both** target episodes under the unknown interface gave 7.12 mm for the unrestricted map and 6.41 mm for the scaled rotation. That spends 30 controls per new interface, whereas the rank-one method can reuse its nominal preparation.

## What the fit could see

The source prior is the mean of the measured native Panda and Sawyer XY response matrices from four balanced, 72-control episodes per arm on Goal 4 and 5, starts 0 and 1. It uses the XY block of each six-dimensional, row-vector OSC gain and multiplies primitive response by five. This approximation ignores cross-axis dynamics, drift and intra-chunk command order. The independent, actively calibrated UR5e matrix is available only in the extra control. [Source episode IDs, ranks and coefficients](../../libero-crossarm-20261120-r1/evidence/gpu01/pilot/responses.json)

For each target arm the reader gets three issued XY means and three world XY displacements from a known nominal episode. It normalizes the source template by one scalar and fits one residual output vector along the nominal input matrix's top right singular vector. The full target baseline fits all four coefficients by least squares on this same episode. Both use nominal π₀.₅ calibration even on the π₀ replication. The shared baseline keeps the source template; a global scalar on that template cancels in the subsequent interface fit.

On a second task, the reader gets three more command–motion pairs at the unknown setting. It fits the two coefficients of a scaled rotation **before** the selected arm-response map. The simplest control uses the identity template. The unrestricted support control directly fits four coefficients to the same three pairs. The scorer then predicts motion from issued commands alone on another task with that persistent setting. The pairing procedure knows which episodes share the interface. The estimator receives neither setting ID nor yaw, gain, grip sign, query motion or success.

LIBERO-10 tasks 5 and 8 alternate as the nominal and unknown-interface calibration tasks. The Goal, Object and Spatial episodes are held out from both roles, with distinct task IDs, seeds and tape hashes. Of eight archived task starts per panel, IIWA Object 1 and Kinova3 Spatial 8 failed historical FK qualification and are excluded. The four common scoring tasks are Goal 3/6, Object 8 and Spatial 0. All passive episodes are at the source archive's layout 11. We use them as cross-task scoring episodes and never access the historical layout-13 retry outcomes. The intervals enumerate the paired bootstrap over four entire fixed tasks, conditional on these arms, one layout per task and the two calibration choices. Generalization across resets or a population of robots remains open. [FK qualification and arm identities](../../symmeta-fk-crossarm-20261123-r1/RESULT.md) · [UR5e π₀.₅ endpoint reconstruction](../../symmeta-camera-command-20261123-r1/WORLD.md)

## Identifiability and cost

Across qualified π₀.₅ traces, the median XY input condition numbers were 10.09 on UR5e, 9.70 on IIWA and 10.68 on Kinova3. Only 0.87–1.05% of command energy lay on the weaker axis at the panel medians. A hypothetical 1 mm perturbation of the support motion array in its worst least-squares direction can change a weak-axis slope by a median 9.0–10.8 mm per unit command. This is a sensitivity calculation. The seven archived Panda active-pulse NPZs had condition number 1.28 for their XY designs. They used scripted pulses and serve only as an excitation reference. With three chunk inputs, a six-input Cartesian Jacobian has at least three unconstrained input directions even if every output component is observed. These XY exports cannot identify a full spatial, contact or gripper map. [Active probe data](../../libero-xemb-joint-sysid-20261119-r1/cpu01/)

| Method | Historical active source controls | Nominal target controls per arm | Unknown target controls per setting | Cold total |
| --- | ---: | ---: | ---: | ---: |
| Scaled rotation or direct unrestricted | 0 | 0 | 15 | 15 |
| Shared Panda/Sawyer | 576 | 0 | 15 | 591 |
| Rank-one target update | 576 | 15 | 15 | 606 |
| Full target nominal | 0 | 15 | 15 | 30 |
| Active UR5e reference | 0 | 288 | 15 | 303 |
| Two unknown-interface supports | 0 | 0 | 30 | 30 |

Counts refer to model-informing controls. Fifteen scheduled controls are 0.75 s at 20 Hz. Across both calibration orderings and all four panels, the analysis reads 96 distinct passive calibration/support tapes (up to 1,440 controls, including the six nominal tapes reused across orderings), 576 source controls, and an optional 288 target-reference controls. This is **2,304 archived calibration controls** including the active comparison, plus 264 separate scored tapes (up to 3,960 controls). The 105 Panda active-design reference controls enter no model. Nominal preparation and source training can be amortized. The historical passive exports lack per-primitive terminal counts for the final chunk.

Acquisition additionally requires a trusted arm model, gripper geometry and a reset joint measurement or reconstruction for the world FK labels on support and score tapes. The upstream audit used a separate 15-control active sibling replay per arm/policy/task reset (32 panel-task checks including two failed cells). Those are qualification controls. They are outside the fit counts, as are simulator settling, reset construction and historical retry costs. This lane used existing exports for every method. It issued zero new environment steps or motor calls.

## Reproduction and receipt

Base revision `757a44cd4374833cad790852df8b87ddf412e638`. The π₀.₅ UR5e export comes from the official frozen π₀.₅ evaluator's `gpu09test` passive archive. The other arm/policy panels include original launch and weight receipts. [Frozen policy loader and official support collection](../../libero-xemb-ur5e-scale-20261120-r1/eval_test_official.py) · [OpenPI pin and policy-specific weight manifests](../../symmeta-fk-crossarm-20261123-r1/QUERY_CONFIG.json)

From the repository root with its uv environment installed:

```sh
OPENBLAS_NUM_THREADS=1 uv run --no-sync python outputs/frozen-vla-factory-20261123-r1/morphology/experiment.py
```

Local Linux CPU, NumPy 2.5.4, job exit 0, 0.42 s elapsed, 44,420 KiB peak RSS, 0 GPUh. Two consecutive runs produced byte-identical metrics. An independent complex scalar least-squares calculation reproduced 6.575295392464616 mm across the 144 primary episodes. [Experiment](experiment.py) · [Metrics, per-task results, design diagnostics and SHA-256 receipts](metrics.json)
