# Frozen Sidney initial-policy diagnostic

Baige Pro6K job `job-on94vctdunlt` completed all80 episodes with zero optimizer updates and released its one GPU at15:38:15 Shanghai on2026-11-06. Source `f1f52cce7dee1bf9285005ba37e47bd819c09061`; the original broad Sidney checkpoint and converted checkpoint/processor hashes remained pinned. Every model stayed frozen. All80 episode files and16 paired initial scene/instruction/RGB/state/exposed-articulation identities were independently reopened;1,851 artifact files totaling1,075,523,792 bytes received independent SHA256 readback. [Integrity](eval01/INITIAL-INTEGRITY.json), [readout](eval01/READOUT.json), [terminal website proof](EVAL01-TERMINAL.jpg).

Use the fixed first16 existing phone evaluation scenes, H400, H50, batch1, same generated episode instructions and matched Gaussian seeds. These are diagnostic counts from a small fixed panel.

| Untouched initial policy condition | Successes | Rate |
| --- | ---: | ---: |
| Original strict-loaded LeRobot, K10 ODE | 3/16 |18.75% |
| Qualified RLinf Sidney port, K10 ODE | 3/16 |18.75% |
| RLinf, K5 ODE evaluation | 1/16 |6.25% |
| RLinf, K5 Flow-SDE0.3, full observations | 5/16 |31.25% |
| Identical Flow-SDE, released fast observations | 5/16 |31.25% |

Native and port K10 agree on14/16 success labels: two common successes, twelve common failures and one discordant success in each direction. Equal aggregate counts do not establish identical closed-loop behavior or remove the separately reported mixed-BF16 drift. The full and fast SDE paths match exactly on all109 saved action arrays, observation-array members and compared state/reward/boundary/sampling records, as well as all16 success labels. This rules out that observation-path difference as the explanation on this panel/batch; it does not qualify arbitrary scenes, hidden physics state or the previous batch64 collector.

Total requested action slots were28,350. The full paths retain measured action-boundary counts; the fast path does not. No measured physical-duration claim is made for this test. It is not a comparison with the final learned step40 endpoints, a six-task benchmark, or a numerical reproduction of Sidney's published66% phone result.

The original LeRobot model here still runs through RLinf's whole-chunk controller and generated prompt. Exact source inspection found that LeRobot's own deployment instead calls native `take_action` per target and defaults to `place phone stand`. RLinf compresses/time-parameterizes the complete chunk and downsamples each arm to at most500 physics points. Those controller/prompt differences need a controlled follow-up before interpreting the low starting competence or spending on longer RL. [Four-condition follow-up](../robotwin-sidney-deployment-diagnostic-20261106-r1/PLAN.md).

The [recipe audit](RECIPE-AUDIT.md) confirms that the completed PPO settings closely follow RLinf's pi0.5 PPO optimization recipe, with the explicitly changed checkpoint/task/horizon/group layout. Our GRPO arm uses the native learner with the shared Sidney PPO-derived settings; the documented released GRPO presets use other motors and differ in optimizer, clipping, filtering and loss granularity. More steps alone remain an untested explanation.

GPU/storage checks passed, host memory peaked32.73769GiB/128GiB, and the output stayed below its12GiB cap. The job charged3,371 scheduler GPU-seconds (worker3,357; conservative creation-to-release3,388). Cumulative actual37.00167 and conservative37.20972 GPU-hours include earlier attempts; the old reservation is closed. A new14,460-second follow-up reservation gives41.22639 conservative worst-case GPU-hours under the unchanged50-hour cap. Required old checkpoints and caches remain protected. [Budget closeout](BUDGET-CLOSEOUT.json).
