# Four-transformer decision experiment

Requested November 12; decision readout targeted for November 15, Shanghai.
The user authorized additional PPO teachers and wants AD, DPT, DIT-style and
IC-IQL transformer results on ID and OOD tasks to decide whether to continue.

## Existing work comes first

The November 5 experiment already trained all four objectives with both KV and
velocity interfaces: eight six-layer, width-256 transformer models, all with
saved validation-selected candidates and complete recovery states. Its source
tasks were OpenDrawer, CloseToasterOvenDoor and CoffeeSetupMug. This differs
from the newer confirmed PPO-teacher set (Drawer, Toaster, Faucet).

The complete native ID/OOD panel exists for AD KV and AD velocity. The remaining
six full evaluations were held near the earlier 150 GPU-hour cap. November 9
started an eight-case KV DPT/DIT deployment pilot. Its completed four banks
were recovered and independently rehashed on November 12; see
[recovered results](RECOVERED-RESULTS.md). The priority is a matched KV AD/DPT/DIT/IC-IQL table,
using the saved models, original cases and original full/reset conditions.
No refitting is needed to answer that first question. Offline losses alone
are not the requested robot results.

The historical cap is retained. Its recorded 140.416111 GPU-hours exclude
later November 9 pilot costs, which must be reconciled before another
allocation is charged to it. Do not silently extend this capped campaign.
Prepare a priced full readout if the remaining allowance cannot cover it.

## Additional source tasks

| Role | Tasks |
| --- | --- |
| Existing confirmed teachers | OpenDrawer, CloseToasterOvenDoor, TurnOnSinkFaucet |
| New PPO candidates | OpenCabinet, SlideDishwasherRack, PickPlaceCounterToStove |
| Held out from the prospective six-task fit | CloseFridge, OpenStandMixerHead, PickPlaceDrawerToCounter |

The new tasks add cabinet, rack and pick-and-place behavior. Native SFT
evaluation already exercised these tasks, but its C5 native sampler and
pretrain scenes do not qualify the C50 RLinf target-scene training protocol.
Run the actual simulator and pre-SGD numerical checks before relying on a
new teacher. Six candidate source tasks do not mean six improved teachers.
Report every candidate, including flat or negative learning.

Start each candidate from the original Human300 SFT, with native first-success
reward, qualified selected-value RLinf runtime, frozen VLM and action-expert
PPO. Preserve historical seed IDs: CounterToStove=6; append Cabinet=9 and
Rack=10. Keep complete chronological observations, executed actions, rewards,
episode boundaries, policy versions, masks and resumable checkpoints. DPT
needs explicit teacher queries at the saved query observations; behavior
actions are not substitutes for missing labels.

The new task budget is at most 50 additional GPU-hours including failures,
qualification, collection, fitting and evaluation. Initially reserve at most
24 for the three bounded teacher pilots and keep the rest for the learner
decision. Full teacher duration depends on measured throughput; do not
promise three successful teachers within that allowance. Check actual queue,
allocation type, storage growth and quota before launch. Never reuse an
output identity or change an active run.

## Readout and interpretation

Use identical transformer architecture and motor interface across objectives.
Compare native task success against the fixed source PPO policies and a
memoryless learner. Include GRPO only after a matched qualified RoboCasa
baseline exists; the existing RoboTwin GRPO numbers are a different benchmark.
Report first-attempt success, success after each real support attempt, paired
full-versus-reset differences, and reward/donor interventions where available.
Keep cumulative best-of-several success separate from next-attempt improvement.

For new fits, split complete scenes/streams before fitting and freeze all
selection on ID development data. OOD means held out from this meta-training
corpus; these atomic tasks may be present in the base VLA's SFT. The historical
three-task OOD panel remains Fridge/Microwave and must not be relabeled as the
new split. Do not mix historical and newly trained policies in a causal
three-versus-six-task comparison.

Continue if native evaluation supports useful history-dependent gains,
including transfer and replicated fits. A competent memoryless policy with
no useful history effect argues against further spending on this recipe.
An underpowered null, weak motor or unfinished rollout is inconclusive about
the general research direction. Show uncertainty and coverage explicitly.

## Current execution

OpenCabinet pilot `job-k7eut34im2s5` was submitted at 14:49:55 Shanghai on
November 12, eight Pro6K GPUs, 64 CPUs, 1024 GiB RAM, 32 GiB shared memory,
60-minute maximum, automatic retries disabled. Its immutable source is
84c982ac2f0fe1c2e18d980d3a0dc3d3eb46b430. The actual eight-GPU allocation
probe passed at 14:50:49. Attempt01 failed at canonical evaluation snapshot creation because the launcher
omitted TASK_CODE_PIN, before a completed PPO update. The fix exports the exact
verified commit. Conservative charge is 0.8023 GPU-hours. Fresh retry and
training progress are recorded in EXECUTION.json.
The other two candidates are configured but not submitted.

The underlying Python change passed all 29 existing flow PPO tests on Baige
Linux with CUDA hidden; all three actual Hydra configurations resolved there.
Local static shell checks passed. Baige development access works through
the existing Jupyter terminal; SSH is not configured on this Mac. Current
shared PFS free space was 273.41 GiB at the runtime inventory, so all three
raw collections cannot be reserved concurrently without verified archiving
or a separate output volume. The existing development allocation has an idle
RTX 4090, 13 CPU cores and 112 GiB memory; it is not an eight-GPU PPO allocation.
The Pro6K console showed 8/416 free GPUs, 22 queued jobs and 32/32 personal
GPUs free. These are snapshots, not launch qualification.
