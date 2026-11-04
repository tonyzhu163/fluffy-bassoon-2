# Four-method source data and held-out evaluation

The five independent PPOs are source learners. Their task-pure histories are
combined for one shared student per method after every source corpus closes
and passes its full audit. Source improvement and student transfer are
separate qualifications. All runs use the original frozen VLA receipt during
student fitting/deployment; a qualified final PPO expert would supply DPT labels.

All five sources have now finished 130 updates. The complete Linux audit and
subsequent hash recheck cover all 650 distinct cohorts, 83,200 episodes and
998,400 decision samples: every camera, continuous target, physical transition,
sampler chain and actor/sampling/environment join passes. However, all five
sources failed both registered improvement gates. They remain reference data;
none is an admitted improving teacher. DPT labels and student fits have not
been generated. See [complete joins](COMPLETE-CORPUS-JOINS-R1.json) and
[native results](RESULT.md). All 80 fixed-batch checkpoint likelihood reports are complete with zero
initial-policy replay error on all five tasks. All five full corpora/QC archives and the latest full checkpoints now pass
[final retention verification](FINAL-CLOSEOUT-R1.json). No fitting or DPT
label queries have started.

## Saved inputs

| Method | Required source information | Retention |
| --- | --- | --- |
| AD | Observation, actual behavior action, feedback, boundaries, learning order | Three raw RGB views, public state25, instruction, executed H50x12 controls, primitive rewards/boundaries, task/seed IDs and producing policy version are saved for every chunk, including failures and post-success physical suffixes. |
| DPT | Same actual context, saved query observation, qualified final teacher | All query observations and full step130 checkpoints are retained. No final teacher passed the improvement gates; offline labels have not been generated. Historical actions/rewards stay unchanged. |
| DIT | Actual actions/rewards and complete physical episodes for value targets | The next saved query supplies nonterminal next observation; all128 horizon observations/update and primitive termination/truncation arrays are retained. Logical PPO success masks are distinct from physical episode ends. |
| In-context IQL | Actual transitions, reward, next observation and physical boundary for Q/V and advantage weights | Same complete physical corpus as DIT, without substituting PPO advantages for the student's fitted values. Behavior log probabilities are also retained for diagnostics. |

Continuous native model actions and processed states are retained in full
sampling records, in addition to decoded physical commands. Bridge targets
must use the qualified native encoding and active-coordinate mask; binary
gripper commands alone cannot reconstruct the original continuous target.
No historical KV edits or velocity-steering codes exist for a full-expert
source, and these four action-supervised methods do not require them.

The production actor archives omit only the approved duplicate camera/chain
fields. Environment records retain all three raw cameras, and sampling records
retain all11flow states, Gaussian parameters and sampler RNG. Actor archives
retain native action targets, tokens/state, old likelihood/value, advantages,
returns, masks and versions. Each Q1's1536actor rows has an audited unique
environment episode/chunk join and byte-exact camera/chain coverage.

`FOUR-METHOD-AUDIT.py` additionally checks native continuous targets, all
physical rewards and suffixes, horizon observations, and exact actor →
sampling → environment joins after deduplication. Native routing removes
environment-only `task_metadata`; its image/state/instruction payload is
otherwise byte-identical. The first audit assumed identical dictionary keys
and stopped read-only; the corrected audit checks that routing explicitly.
R3 runs independent task audits concurrently within the allocated CPU quota;
the sequential R2 was stopped without simulator calls or SGD.

All 130 versions per task and their numerical corpus joins are now verified.
The failed source improvement gates prevent teacher admission under this
contract. Any later fitting plan must address that scientific result, query
qualified DPT labels and qualify each reactive adapter before history fits.
The existing direct
exporter is hard-coded to the old18-update source and needs adapting to this
campaign; data readiness does not mean the new fitting pipeline is complete.
The saved decision RGB is sufficient for these four students; it is not a
claim that full-rate episode videos or arbitrary future reward-model inputs
were recorded.

The [completed allocated-CPU audit](FOUR-METHOD-AUDIT-R3.json) passed all
five Q1 corpora and the first production update of OpenDrawer, Toaster and
Drawer-to-Counter:1,024episodes/12,288chunk records. All native continuous
targets, physical rewards/suffixes, horizon observations and unique
actor/sampling/environment joins pass. This samples the running campaign;
this was an early qualification; complete coverage is recorded above.

## Held-out evaluation

Student training uses only OpenDrawer, CloseToasterOvenDoor,
PickPlaceDrawerToCounter, CoffeeSetupMug and TurnOffStove. The adopted held-out
tasks are CloseFridge, OpenStandMixerHead, PickPlaceCounterToStove,
TurnOnMicrowave and TurnOnSinkFaucet, at their declared native horizons.
They are held out from this source/meta-training campaign; this does not
claim they were absent from the original VLA's pretraining.

Each frozen student starts with empty memory on each new target trial and
collects its own support attempts/rewards, followed by separate query cases.
Pair cases/seeds across methods, report success versus actual target controls,
and include original motor, memory-reset and no-history/reactive controls.
Do not select a source expert using held-out test outcomes.

The five task-specific source PPOs do not supply an online PPO baseline on
the held-out tasks. The proposed comparator is independent online PPO from
the same original motor on each target, with target interactions, updates,
wall time and GPU cost reported separately. A10Mtarget-control endpoint is
an eventual-performance reference, not an interaction-matched comparison to
a few support attempts. Target PPO runs and any GRPO comparator are separate
experiments, not launched by this source campaign. Small-budget PPO settings
must be qualified explicitly rather than silently inheriting the source's
128-episode update batch.

The [production-only R4 audit](FOUR-METHOD-AUDIT-R4.json) also passed Coffee
and Stove after their allocations started. Combined R3/R4 qualification now
covers every task's Q1 and first resumed production batch:1,280episodes and
15,360chunks. The later full 650-cohort audit supersedes that sampled coverage.
