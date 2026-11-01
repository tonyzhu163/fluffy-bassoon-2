# Native fixed100 comparison — prepared, not admitted or launched

The source owner requested this code/plan preparation after independently
auditing all four real first-fit models. No evaluation allocation, ledger
reservation, simulator reset or robot action has been launched here.

## Exact panel and case bank

| Condition | Weights / initial-noise convention | Episodes | Maximum actions |
| --- | --- | ---: | ---: |
| Native Gaussian | Frozen Motus, unclipped standard Gaussian noise | 16 | 6,400 |
| Native zero | Same motor, zero initial noise | 16 | 6,400 |
| SAC5 mean | Fixed source SAC5, deterministic physical1.5*tanh(mean) | 16 | 6,400 |
| AD full | Fixed100 portable endpoint, deterministic clipped mean | 16 | 6,400 |
| AD static | Independently trained static fixed100 endpoint | 16 | 6,400 |
| DPT full | Fixed100 portable endpoint, deterministic clipped mean | 16 | 6,400 |
| DPT static | Independently trained static fixed100 endpoint | 16 | 6,400 |
| AD full reset | SAME AD full weights, erase memory/feedback each episode | 16 | 6,400 |
| DPT full reset | SAME DPT full weights, erase memory/feedback each episode | 16 | 6,400 |
| **Total** | **9 conditions × 2 targets × K8** | **144** | **57,600** |

Tasks remain only `adjust_bottle` and `move_stapler_pad`, each400 primitive
actions per episode. There are18 complete K8 units. Every condition uses the
same eight requested seeds PER task:

`22261030, 22262027, 22263024, 22264021, 22265018, 22266015, 22267012, 22268009`.

These are the existing development-target repetition0 seeds:
`20261030 + 2,000,000 + episode_index*997`. They are not the old native-before/
after bank. Native Gaussian must run first on each task to establish a new,
complete paired reference. Actual accepted seeds, raw initial-observation
hashes and feature hashes must match across all conditions, or the unit stops
before its next motor query. Planning retries and reset failures remain logged;
reset-internal physics is not policy-action accounting and remains separately
unmeasured where the qualified runtime has no counter.

The primary reported episode positions are1/2/4/8. This is K8 TOTAL, not eight
support episodes plus another query set. Any descriptive support/query split
uses existing episodes1–4 and5–8 and collects nothing extra. One trial per
target/condition provides a small paired development screen, not a statistical
demonstration of ICRL. No target score chooses endpoints or action conventions.

## Fixed artifact provenance

All four real models finished100 updates in Baige `job-ielg6papl3hn`, model
code `1aeea4f9fcc307bb147def92374bb81a3e2cb50e`; launcher
`41c711c1e61d3311b4c7f022cf3b3e0f1fddb5d7`. Each saw39,621 supervised
positions from1,539 unique source decisions. Source-owner finite/CRC/Adam/
sampler/RNG recovery audits passed. The first-fit summary was read back
independently, including `completed_all_four` and exit0; its exact SHA is in
`artifact-inputs.json` and `plan.json`.

Fixed portable endpoints under
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-firstfit-20261030-r1/`:

| Endpoint | SHA256 |
| --- | --- |
| `fit-ad-full/candidate-000100.pt` | `02896a84a880c1ec79d1b8f2e574cd74ee5e456ec55c6280bfdc0c9906aba7b6` |
| `fit-ad-static/candidate-000100.pt` | `bf7a0e00483948853402e1b3440f01779930ce93a9a20638608ed67ab696da4f` |
| `fit-dpt-full/candidate-000100.pt` | `241b20579fc5ba76049ce0772770d1f57f0ba437ef5b8933cad5be365e30d05f` |
| `fit-dpt-static/candidate-000100.pt` | `cba9ac8dabc59db0615142302e309ca4d74320eef81b17d0b33ba4eed6e5a9ed` |

SAC5 teacher:
`/mowenjie/fluffy-bassoon-2/exports/robotwin-native-prefix-20261030-r1/sac-source-recovery.pt`,
SHA `f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42`.
The teacher mean is a source-trained candidate whose competence remains a
measured outcome, not an assumed expert label.

Train index SHA:
`4e8d4e64eb1845a234f710342f168d3a4cdb86366d64dd6c7cef6c7c5895a4af`.
Validation index SHA:
`194ae0fd4588e7bd55c3cd7a0fc6eedfb010bcb8043e311eeedd8adff0b2472b`.
These bind source-training normalization and the SAME source SAC5 DPT teacher.
The selected training bank contains47,602 source actions; the separate PPO10
source-validation panel contains10,552 validation actions and depends on its
93,851-action source anchor. These already-charged campaign costs are reused
once across the four models; this plan does not recollect or re-import them
into each condition's ledger. The57,600 cap is NEW evaluation allowance only.

## Qualified runtime and preparation status

The runtime must use native backend from
`db0184089fa88a5f96c153f9b80029bb91b77def`, file
`src/fluffy_bassoon/rl/robotwin_noise.py`, SHA
`eacc64b4daf8f78aeb5e023a2c4d2a2a040eacfd2e662335c118334c2ab50ff3`.
The adapter checks this byte identity and the complete scientific runtime
contract, excluding only allocation-specific render PCI. Norm stats, assets,
Motus frozen fingerprint, native action transforms, H32/full1024, final-camera
capture, source features and per-task limits remain pinned in `plan.json`.
The backend inherited by the old trainer checkout is incompatible and fails
this check. Do not select it by PYTHONPATH order.

Prepare a clean integrated execution commit by reviewing/cherry-picking the
native history changes and this adapter onto the qualified runtime, retaining
the backend bytes. Record the integrated commit, training commit and backend
blob hash separately. Latest main must be reviewed again before launch; this
preparation's fetch attempt could not refresh main because Git SSH access was
unavailable. No active source checkout is edited or silently migrated.

Code supplies `admit_plan`, `load_agent`, and `run_trial`; the execution owner
retains simulator construction, GPU admission, parent process lifetime and
wall supervision. See `docs/robotwin-history-evaluation.md` for the exact call
boundary. Pass raw `RobotwinNoiseEnv` and the existing `CampaignBudget` once;
the adapter supplies the sole pre-dispatch chunk reservation.

This candidate passed static syntax/whitespace inspection ONLY. Prepared
allocated-CPU tests cover fixed endpoints, byte-identity/runtime rejection,
paired accepted cases, native Gaussian versus zero, pre-dispatch accounting,
replay refusal, unknown outcomes and honest zero-action terminal handling.
One controller lifecycle check was widened to admit a native zero-action
terminal outcome; live nonterminal zero-progress still fails. The trainer and
saved checkpoint files remain untouched. These changes require allocated CPU qualification and
the integrated runtime gate before an evaluation launch.

## Exact proposed cost ceiling, not a reservation

Suggested CPU qualification:0 GPUs,4 CPUs,16 GiB, at most300 seconds.
This is at most1,200 CPU-seconds and zero robot actions. No allocation has
been requested. Relevant tests are `tests/test_robotwin_history_evaluate.py`,
`tests/test_robotwin_history.py`, plus existing native environment/budget/corpus
tests on the clean integrated owner pin.

Suggested robot panel envelope: **one96GB GPU,16 CPUs,128 GiB RAM,8 GiB shared
memory, at most5,400 GPU-seconds /90minutes**, including all startup, probes,
model loading, parity calls, failures, finalization and any approved retry.
This is at most24 CPU-core-hours and192 GiB-hours. Reserve at least5 GiB of
project output capacity before launch. These are proposed ceilings for owner
review, not spending already admitted here.

At previously observed scalar collection rates28.62–32.18 actions/s,57,600
actions imply about30–34 minutes of collection. The older18.95 actions/s
reference implies50.7 minutes. These are planning estimates from different
workloads, not guaranteed evaluation throughput. Nine motor loads, reset and
renderer lifetimes, raw capture, hashing and failed attempts consume the
remaining headroom and must be measured. Motor actions are the hard physical
cap; actual queries and GPU wall time have independent counters.

At full nonterminal H32 chunks,400-step episodes need13 motor queries, so
1,872 is the nominal panel query count. Valid partial nonterminal chunks could
increase it; record actual query counts instead of treating this as a hard
query cap. Parity inference and setup also consume motor/GPU work with zero
physical actions. Record accepted resets, failed attempts, actual renderer
scene creations, raw frames, model/motor queries and per-condition timing.

Use nine sequential fresh processes, one condition each, with at most16
accepted episode resets per process. Preserve a complete K8 result before
moving to the next task or condition. This is well below the observed roughly
1,000-reset failure lifetime but does not prove native lifetime stability.
The execution owner guards cumulative time before each whole trial, retains
partial raw/ledger evidence on any fault and reports missing conditions if the
cap binds. No lost physics state is replayed as though it were a free resume.

The owner creates a new separately metered parent/child evaluation allowance
and exact ledger contract before launching. No source cap, old optimization
hold, trained recovery, native reference or fixed100 endpoint is replaced.
There is no tenth condition or wider grid in this proposal.

Keep `plan.json` immutable. Guard its file SHA during transfer and use its
canonical JSON SHA in the ledger and native case receipts. Record admission,
execution status, allocation IDs and real costs in separate run artifacts;
changing the plan's status field would also change its pairing identity.
