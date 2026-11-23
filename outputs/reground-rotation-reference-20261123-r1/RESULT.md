# ReGround’s published 90° rotation condition and streaming SymMeta

The released ReGround controller recovered a success IQM of **0.974** on the rotated Panda, close to the **0.98** reported in Table 1. The frozen policy scored zero on every test episode. Streaming SymMeta recovered some tasks without a separate target fit, with success IQM **0.186**. ReGround’s unsafe-episode IQM was **0.339**, above the published **0.29**. Its largest task-level success discrepancy was the long-horizon drawer task, **0.577** here versus **0.77** published. The reference success result transfers much more closely than its unsafe estimate or hardest-task result. [Published Tables 1 and 2](https://arxiv.org/html/2611.20244v1#S4) · [Released scorer output](evidence/native-release-summary.json)

| Method | Successes / 1,400 | Unsafe / 1,400 | Success IQM, release 95% CI | Unsafe IQM, release 95% CI | Start-cluster 95% CI, success / unsafe |
| --- | ---: | ---: | --- | --- | --- |
| Frozen policy | 0 | 1005 | 0.000 (0.000–0.000) | 0.843 (0.817–0.869) | 0.000–0.000 / 0.786–0.900 |
| Streaming SymMeta | 401 | 375 | 0.186 (0.160–0.207) | 0.200 (0.167–0.219) | 0.136–0.243 / 0.136–0.257 |
| Fitted ReGround | 1271 | 582 | 0.974 (0.957–0.991) | 0.339 (0.296–0.390) | 0.966–0.987 / 0.306–0.376 |

All 4,200 scheduled episodes are included. The published ReGround success point lies inside both rerun intervals. The published unsafe point lies below both. The authors report rounded estimates and the intervals here describe rerun uncertainty. For frozen Base, the reported unsafe point of 0.80 lies inside the start-cluster interval. [Full comparison and paired uncertainty](evidence/comparison.json) · [Original case rows](evidence/native-cases.json.gz)

ReGround succeeded on 873 assignments where SymMeta failed, and SymMeta succeeded on three where ReGround failed. The arithmetic mean success difference was **+62.14 percentage points** for ReGround (95% start-cluster interval +58.21 to +65.86). ReGround also had **14.79 points more unsafe episodes** (+10.43 to +19.21). Of 1,271 ReGround successes, 522 were also flagged unsafe. For SymMeta that count was 30 of 401. Lower unsafe incidence accompanies much lower task completion here. [Paired contrasts](evidence/comparison.json) · [Unsafe components and joint outcomes by task](evidence/outcomes-by-task.json)

## Success by task

Each task contributes 35 held-out starts under five seed assignments. The table compares per-task arithmetic means with the corresponding published values.

| LIBERO task | Published ReGround success | ReGround successes | SymMeta successes | Frozen successes |
| --- | ---: | ---: | ---: | ---: |
| spatial_0 | 0.99 | 175/175 (1.000) | 80/175 | 0/175 |
| spatial_2 | 1.00 | 173/175 (0.989) | 15/175 | 0/175 |
| object_1 | 1.00 | 167/175 (0.954) | 90/175 | 0/175 |
| object_3 | 0.99 | 172/175 (0.983) | 20/175 | 0/175 |
| goal_1 | 0.99 | 174/175 (0.994) | 171/175 | 0/175 |
| goal_2 | 0.95 | 163/175 (0.931) | 0/175 | 0/175 |
| 10_1 | 0.88 | 146/175 (0.834) | 15/175 | 0/175 |
| 10_3 | 0.77 | 101/175 (0.577) | 10/175 | 0/175 |

## Opening actions and controlled rerun

All three methods agreed on all 1,400 initial robot states, task assignments and policy-noise keys. The original strict three-way check stopped at long task 3, start 44. Seven assignments across two shards differed in their first proposed actions, before the streaming reader intervened. Six had different archived agent-camera images on one streaming seed. The simulator can move a fixed scene fixture when a worker is reused. The original Base images for the seventh case are unavailable. On the 1,395 exact SymMeta/ReGround openings their success counts were 401 and 1,267. Allowing arbitrary binary outcomes on the five unresolved pairs bounds the full-grid arithmetic mean advantage for ReGround at 61.50–62.21 points. [Strict check output](evidence/original-strict-readback.log.gz) · [All pairwise bounds](evidence/comparison.json) · [Reset investigation](prefix-audit/RESULT.md)

A separate controlled rerun covered both affected shards in full, including their 25 companion assignments. All three methods received the same fresh native camera inputs, simulator state, policy noise and full 50-action opening proposal. The counts on those 32 assignments were **0 / 12 / 31** successes for Base / SymMeta / ReGround, compared with **0 / 13 / 31** in the originals. Unsafe counts were **31 / 4 / 9** in both runs. One streaming companion changed success after a later policy query. ReGround exchanged one success for one failure across the seven discrepant starts. The original scores above retain all rows as run. [Independent audit and individual reruns](common-opening/readback/RESULT.md)

## What each method saw

The eight LIBERO tasks were evaluated on Panda, with the released 90° rotation applied to both the translational and angular action deltas. Each method used one continuous attempt, the same frozen π₀.₅ weights, the same policy-noise rule, four native environments per batch and a new 50-control proposal every ten executed controls. The five adaptation seeds each used the 35 held-out starts (15–49) for each task. This gives 1,400 assignments per method and 280 distinct task–start pairs. In the two frozen arms a seed label repeats the same policy and reader. In ReGround each seed has its own fit. The authors’ historical checkpoint hash is unreported. We strictly loaded the accessible LeRobot checkpoint at `7152f13e34c6c115d808aea0e5d08aca74eac14b` against ReGround release `16a98486f560676b9036d1a6d067b9dd51a5917a`. [Pinned release and schedule](REFERENCE-PROTOCOL.md) · [Runtime, weights and simulator qualification](RUNTIME.md)

ReGround used the released 25-dimensional robot state, inverse-dynamics anchor, CEM horizon 1, four iterations and 256 candidates. The optional shift monitor was disabled, as configured for this cell. We collected 1,041,714 motor-babbling transitions (5,409 complete episodes) and 1,140,866 policy-driven transitions (5,440 complete episodes) across the nominal and rotated bodies and the five seeds. Each fit used a 100,000-transition nominal subset (60,000 babbling, 40,000 policy-driven) and a 10,000-transition rotated subset (6,000 and 4,000). The nominal and deployed forward and inverse models reuse the respective subsets. Each fit is shared across all 280 of that seed’s test cases. We archived all 20 model hashes, training metrics and input counts. [Motor-babbling shards](evidence/native-babble-manifest.json.gz) · [Policy-driven shards](evidence/native-policy-manifest.json.gz) · [Model receipts](evidence/fitted-models.json)

Frozen SymMeta used zero separate target calibration transitions, zero extra resets and zero probe controls. The seed-41 reader estimated the interface from public fixed-camera images, issued commands, tool position and jaw measurements during the scored episode. It began at identity, could change the 11th action, and updated on five-control chunks for up to 120 controls. It corrected planar translation, translational gain and gripper sign and passed wrist commands through. The released shift also rotates wrist commands. The arms therefore differ in target calibration data, correction family and when the correction can first act. [Reader and timing](MATCHED.md)

## Scoring and uncertainty

Each episode contributed its native LIBERO task success. The released unsafe flag is the union of non-target contact, displacement, falling and exceeding the released speed envelope. Success and unsafe can occur in the same episode. Each published IQM trims the lowest and highest ten of the 40 task-by-seed rates and averages the middle 20. The release’s 95% interval resamples seeds within each fixed task and episodes within each sampled seed. The start-cluster interval resamples 35 starts within task with all five outcomes retained together, conditional on these tasks and fits. The paired differences use this same start-cluster procedure for the arithmetic mean. Both interval procedures use 10,000 draws. [Released statistics](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/eval/stats.py) · [Cross-arm scoring and sampling](compare.py)

## Consequence for the paper

The comparison supports a short-trace visual calibration result for the researcher-imposed, second-attempt protocol. On the released one-attempt 90° condition, a target-fitted controller recovers much more task success. The paper's 0.75-second support result, the native streaming result and the target-fitted reference answer different deployment questions. In the separate same-source retry experiment, the learned visual readout has a measurable advantage over a six-coefficient projection, primarily on novel combinations. A tuned Perceiver also recovers held-out yaw with a direct readout. These observations limit architectural attribution. [Matched-source experiment](../symmeta-matched-flow-20261123-r1/RESULT.md) · [Tuned Perceiver comparison](../symmeta-wider-family-20261123-r1/RESULT.md) · [Revised manuscript](../../papers/icml-xemb-sysid/main.pdf)

All five fits and three native evaluator jobs completed on Baige Distributed Training. The native ReGround evaluators were `job-hgy24rsvpfdb`, `job-9jl2fuv06v5t` and `job-eg2n3i57xvtd`. The two studies used **33.02 GPU-hours** of measured launcher and qualification time, including failed attempts and supplemental reruns. The full traces, fit caches, calibration pools and batch checkpoints remain on `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1/`. [Allocation ledger](evidence/ledger.json) · [Execution record](EXECUTION.md) · [Completed evaluation jobs](evidence/completed-native-evaluations.jpg)
