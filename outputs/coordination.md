# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-10-31 Asia/Shanghai**.
Keep current decisions and jobs here, method contracts in `docs/`, verified
results in `EVALUATION.md`, and dated plans and evidence in
`outputs/<run-id>/`. Follow [AGENTS.md](../AGENTS.md) and the
[retention policy](../README.md#artifact-retention).

Earlier state, including the fallback ideas, run order, DSRL how-to, CMM
calibration table and completed 2026-10-28/29 jobs, is in the
[2026-10-31 snapshot](archive/coordination/coordination-20261031.md). The
[cutover placeholder](archive/coordination/coordination-pre-cutover-20261028.md)
and [source-repository records](https://github.com/tonyzhu163/fluffy-bassoon/blob/lycoreco_rl/outputs/coordination.md)
are older.

## Question and decision

Can a persistent CTM learn from interaction history with fixed deployment
weights, and does its recurrence offer an advantage over conventional memory?
Separate architecture performance from motor competence, steering headroom,
informative history and optimization.

- If XL adapts and CTM does not, retain the qualified pipeline and use the
  stronger learner while revising the CTM claim.
- If neither adapts, diagnose the interface, histories and training signal
  before attributing failure to memory.
- Judge adaptation with paired fresh probes, trained static/reset controls
  and valid history interventions.

Contracts: [support/probe](../docs/ctm-support-probe.md),
[feature bridge](../docs/pi05-feature-bridge.md),
[PPO replay](../docs/pi05-bridge-policy.md).
Results: [meta-RL evaluation](../src/fluffy_bassoon/meta_rl/EVALUATION.md).

### Redesign (2026-10-31)

No robot comparison so far shows memory or adaptation. The governing plan is
the [frozen pi0.5 steering screen](frozen-pi05-steering-screen-20261031-r1/PLAN.md):
first show a controller can improve the frozen motor on cases it fails, then
show experience changes a useful later decision. The
[steering-headroom audit](steering-headroom-audit-20261031-r1/AUDIT.md) gives
the paired fixed100 evidence. The earlier
[headroom diagnosis](archive/icrl-headroom-diagnosis-20261031-r1/PLAN.md) is
archived as superseded, with corrections listed at its top. A
[criteria proposal](steering-interface-criteria-20261031-r1/PROPOSAL.md)
informs the newly authorized [targeted scope](frozen-pi05-endtoend-20261031-r1/SCOPE.json),
which adopts A1/A2/A3 with the [review safeguards](frozen-pi05-steering-screen-20261031-r1/CRITERIA-REVIEW.md).
The proposal is not adopted verbatim: search is explicitly bounded, improvement
must be measured rather than assumed, and the initial mechanism panel does not
replace fresh32-case confirmation before broader claims.

## Coordination rules in force

From the [2026-10-31 review](archive/coordination/coordination-20261031.md#coordination-review-2026-10-31):

- Push coordination edits to `main`, rebased, editing only your own row.
  Branch-only edits leave this file stale.
- One or two readable sentences per row: job ID, last check, progress, next
  action and a link. Put hashes, byte counts and budget ledgers in the
  linked run record.
- Budget: roughly **1M source actions per learner** (human, 2026-10-30;
  confirmed 2026-10-31), superseding "1M per benchmark, shared". Charge each
  learner's actions to its own ledger; do not borrow across learners or
  benchmarks.
- Baseline priority (human, 2026-10-31): same-motor DSRL/SAC first, native
  VLRLA PPO afterward. `frozen-vla-poc` (`finish_assembly`/`finish_ppo`) is
  the sole LIBERO launcher; RoboTwin AD GPU work stays deferred.
- Active `codex/*` branches were merged into `main` on 2026-10-31; merge
  further verified work promptly (see [AGENTS.md](../AGENTS.md)).

## Ongoing work

Rows are as last recorded by their owners; none was re-queried for this
update.

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Original Pi0 DSRL task57 replication | Owner chat `01a192ca-ebbb-7970-827d-fdf784722bc9`; Baige GPU `job-psgxs9n5b08w`, source `00eca3fa`, seed0 | **2026-11-01 04:47 Shanghai direct audit:** Healthy at 238,080 published updates, 1289 episodes, 12404 replay query rows. Native 4/10; first learned 10/10 at 40k, 16 perfect panels through 190k; 200k 9/10 then 210k–230k 10/10; latest 230k scored 10/10 at 12000 rows/233,512 training primitive actions. Full recovery 232,180 and 24 evaluation payloads SHA256-verified; model C+G GPU contexts verified. R4 output 4.05GB; ledger 288,812 known +400 active pending +400 prior =289,612 phase actions. Conservative live GPU upper bound 9,987s includes prior 660s, under original 36,000s/1M actions/64GiB caps. Full run/final motor hash/three-seed aggregate pending. 20min monitor active; next audit recovery and stability, quiet for unchanged healthy performance. |
| Native VLRLA-recipe PPO pilot | Run `vlrla-native-ppo-pilot-20261031-r1`; owner `frozen-vla-poc` | **2026-10-31 14:06 Shanghai:** queued behind the task57 DSRL run; native CPU qualification closed, GPU held. Specified at 6 GPU-hours / 1M actions. |
| LIBERO task44 DSRL on frozen pi0.5 | Baige `job-w8deahieli8q` Running; sole `finish_assembly` | **Nov1 04:40:**154920updates/799episodes,760success39failure;completed140080panel20/20,1511+200actions;early20580=1/20/alllaterpanels retained. Stable laterfull155080,newraw/RPC,finite losses;219697phaseactions/11.11GB;fixed500k/finalnative pending,next05:00. [Proof](libero90-pi05-headroom-runtime-20261101-r1/CHECK-0440-PROOF.json). |
| Frozen pi0.5 interface/data redesign | Cache04 and readoutCPU terminalFailed; sourceABI correction CPU pending; sole `finish_assembly` | All13encoding/RNG closed,paid11 unchanged;3row CPU inspection isolates officialstate(1,32) vsconsumer(1,14),others pass. GPUactual3625/actions23922;originalCPU1775/1800,preservation98/120,readout69/120(56PASS1fixtureFAIL),no oldcap reset. New300CPU correction12+readout1(+optionalbridge3),then independent all90CPU seal;noGPUreencode/fit/source. [Measured receipt](frozen-vla-readout-software-qual-20261101-r1/RECEIPT.json). |
| LIBERO90 pi0.5 headroom screen for DSRL replication | Baige `job-cth8mugcqgzl` terminal Succeeded/exit0; immutable `4ec35eb9`; owner chat `01a192cf` | **Nov1 02:54 saved-evidence review:** own screen audited 384/384, task59 native3/16 vs untrained tied9/16; motor unchanged,3,890GPU-s/134,629actions. Separate task44 SAC has interim20/20 at40,140 updates after1/20 at20,580; final500k/native/persistence/motor pending. Conditional interface reconciliation prepared; no duplicate learner or interface GPU launch. [Gate and task distinction](libero90-pi05-headroom-screen-20261101-r1/REFERENCE-RECONCILIATION.md). |
| RoboTwin final100 AD export (R4) | Baige `job-vth2qv6r4lxu`, CPU only | **2026-10-31 13:38 Shanghai: complete, exit 0.** AD train and BC validation exports pass independent audits; no new GPU, robot actions or SGD. AD GPU fit stays deferred for the DSRL/SAC-first priority. [Final report](robotwin-final100-ad-export-20261031-r4/RESULTS.md); [source campaign](robotwin-trained-continuation-20261031-r1/FINAL-CAMPAIGN-RESULTS.md). |
| Meta-World released pi0.5 noise screen | Baige `job-orgyjb0z9h6d`; run `metaworld-pi05-noise-baige-20261029-r2` | **2026-10-29 20:17 Shanghai: paused at user request.** Reach screens fail headroom; goals 1 and 3 never reached. Remain paused. [Pause record](metaworld-pi05-noise-baige-20261029-r2/PAUSED.md). |

### Needs owner refresh

Last recorded 2026-10-28 19:23 Shanghai; state since then unknown. Owners:
confirm terminal state and move each to completed work.

- RoboTwin GRPO learning gate, Baige `job-vh9lp7fvgf9n`: initializing, one
  GPU / 20-minute cap.
- MetaWorld bounded pilot r2, Baige `job-x2x4kdpfn011`: first evaluation
  saved, one GPU / 180-minute cap.
- CMM decay diagnostics thu2 48983/48984: last recorded queued. The packed
  diagnostic 49178 below covers the same comparison; confirm these were
  cancelled.

## Recently completed work

Details for each are in the [snapshot](archive/coordination/coordination-20261031.md).

- **CMM calibration**, thu2 48876: complete, failed (4.55% vs paper 30.63%);
  optimization instability. Packed decay diagnostic thu2 49178: no clear
  speedup. CMM structure tests stay gated on calibration.
- **DSRL Stage 0 smoke**, Baige `job-gjkse1sb5gyj`: repeat determinism failed
  (4/32 identical trajectories); S0b/S0c not run.
- **Native-context pilot r3**, Baige `job-8kkjhb65px67`: failed feedback
  validation, exit 1; preserved.
- **Native pi0.5 CTM/XL PPO, support/probe gate, privileged tracker**:
  complete and audited 2026-10-28; no ICRL or architecture claim.
- **Cached CTM/XL/GTrXL, CMM cached screen, DPT warm starts**: complete; one
  cached seed, no CTM advantage.
- **Baige environment cleanup**: complete 2026-10-29.
