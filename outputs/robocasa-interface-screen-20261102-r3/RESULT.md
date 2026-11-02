# Stage 0 interface screen: completed, no PPO training

Baige `job-8ppjn68jxa9e` succeeded on November 2 at 20:08:06 Shanghai, using
immutable source `d126c0a2c07984dfac2a0151c1506bf9e374a2b6`.
All six planned families were attempted. Four perturb the existing expert;
one replaces its action outputs with a distilled head; one selects candidates.
The replacement head must not be ranked as frozen-expert steering.

The four expert-perturbation families show finite recovery but no clear positive
per-task advantage over matched native resampling in this screen. This is not a
PPO learning experiment and cannot decide which interface PPO can improve.

## Completed expert-perturbation panels

Each entry is successes /160: 20 cases selected by a separate failed native
anchor, with eight subsequent draws per case. These are selected TRAIN cases,
not an unrestricted or held-out evaluation. Every episode has 600 policy
controls, with no success termination.

| Family / variant | OpenDrawer | DrawerToCounter | Coffee | Stove |
| --- | ---: | ---: | ---: | ---: |
| Native resampling control | 79 | 43 | 44 | 16 |
| Action residual 0.25 | 81 | 42 | 31 | 12 |
| Action residual 0.5 | 42 | 38 | 16 | 7 |
| Bounded tied initial noise, 32-D | 1 | 0 | 0 | 0 |
| Conditioning / KV value residual | 89 | 44 | 47 | 20 |
| Denoiser velocity residual | 84 | 49 | 53 | 20 |

Native best-of-eight recovers 19/20, 15/20, 16/20 and 9/20 selected failures.
That uses simulator truth to choose among eight complete episodes. It is
oracle search headroom, not a deployable selector or an upper bound on all
chunkwise policies.

The full per-case outcomes, paired wins/losses, exact best-of-eight tests and
rate tests that swap all eight draws together within each case are in
[FINAL-RESULT.json](FINAL-RESULT.json). These are exploratory, unadjusted tests.
Positive per-task steering differences do not reach p<.05. Stronger action
residuals and this particular bounded tied-noise distribution often degrade
performance; that does not rule out other scales or learned noise steering.

Conditioning changes final-layer KV values through a fixed projection;
velocity adds a bounded code at each denoising step. The amplitudes were fixed
before outcomes. Both reproduce the native path at zero control. Neither is
a trained bridge or history-conditioned controller here.

## The other two planned families

**Direct action head: distillation diagnostic, not expert steering.** A
2064→256→600 MLP maps frozen prefix features and public state to a full H50×12
action chunk. It excludes the explicit native-action features, imitates saved
native chunks, and executes its predictions with a 6-D action residual.
It bypasses the original expert's action outputs; the current runtime still
computes native features/actions, so this is not a measured speedup.

OpenDrawer passed the eight fresh TRAIN competence cases: head 8/8 versus
native 3/8. Its subsequent full panel scored 109/160 and recovered 20/20 cases,
versus native 79/160 and 19/20. **Those same target cases contributed imitation
training/validation data and model selection. This is not an independent
steering or generalization result, and possible overfitting remains unresolved.**
The eight-case competence check is too small to establish broad generalization.

The head was attempted but remains unqualified on the other three tasks:

| Task | Qualification outcome |
| --- | --- |
| DrawerToCounter | Task validation MSE .592 exceeds .25; no head rollout admitted |
| Coffee | Fit passed; direct 0/8 versus native 3/8, competence failed |
| Stove | Fit passed; direct 0/8 versus native 0/8, positive-success gate failed |

Failed admission leaves the family incomplete; it is not evidence against
all direct-head PPO. [Data split, weights, Adam and RNG audit](DIRECT-AUDIT.json).

**Eight-candidate selector: completed on all four tasks.** It executes one
rollout per case and scores eight candidates at each chunk with a frozen
source-PPO value heuristic. This scorer is not a qualified Q-function and has
no adaptation memory.

| One-rollout policy | OpenDrawer /20 | DrawerToCounter /20 | Coffee /20 | Stove /20 |
| --- | ---: | ---: | ---: | ---: |
| Matched native draw 0 | 12 | 4 | 4 | 3 |
| Frozen value selector | 13 | 5 | 4 | 1 |
| Selector paired wins/losses | 5/4 | 4/3 | 3/3 | 1/3 |
| Exact paired p | 1 | 1 | 1 | .625 |

No clear selector benefit is established. Each selector panel costs 1,920
native candidate draws and 12,000 physical policy controls. Native eight-draw
search uses the same 1,920 motor draws but 96,000 physical policy controls and
oracle truth selection. Those are different decision procedures.

## Implication for the steering question

The human clarified that retaining and steering the frozen action expert is
the intended question. The direct-head diagnostic is therefore separated from
the steering comparisons. VLRLA-bridge retains the action expert and adapts KV
conditioning; this direct head is closer to a memoryless replacement-policy
baseline, not VLRLA-bridge. [VLRLA §4.1](https://arxiv.org/pdf/2610.19517).

The two families beyond the four perturbations need different preparation:
the direct head needs a competent imitation initialization, which qualified
only on OpenDrawer; selection retains the motor but needs a useful
candidate-aware scorer. This existing V-value heuristic is not evidence that
such a scorer has been learned. No new scorer training was launched.

Conditioning/velocity adapters can begin at zero-output identity and preserve
the native expert. Whether PPO can learn useful departures, and whether a
separate adapter warm-up helps, remain Stage 1 questions. Actual improving
simulator- and Flora-reward PPO curves must precede new useful AD/DPT history
collection; no Stage 1 or PPO launch was authorized here.

## Pipeline check, integrity and costs

The complete owner-published AD/DPT pipeline readout was independently
recalculated on Linux CPU. ID full/static counts are AD 34/29 and DPT 35/30
out of 50, with exact paired p=.125 for both. OOD counts are AD 33/33 and DPT
27/26; neither establishes a memory advantage. K1-versus-K0 comparisons also
remain inconclusive. These use their separate 750/900 horizons, not this
screen's 600. [Complete pipeline check](PIPELINE-CHECK.json).

All 35 completed banks passed a fresh final CPU audit of plans, actual chunk
records, controls, truth/video/checkpoint hashes and exact physical/public
pairing. Original camera pixels remain unchanged; RGB differences are
recorded diagnostically. The 349 carried outcomes were reverified, with no
case/draw changes. The 213 carried native outcomes lack actual chunk records
and must not be treated as AD/DPT training histories. The current attempt
retains 3,915 complete episode checkpoints and 46,980 actual chunks.
[Final bank audit](FINAL-BANK-AUDIT.json), [carry recheck](FINAL-CARRY-RECHECK.json).

The complete campaign has 4,264 admitted episodes and 2,558,400 complete policy
controls; 2,582,096 controls are charged including partial failures and
conservative unresolved intents. Sixteen old unresolved calls remain charged.
Native motor draws total 58,760, including 416 calibration draws and all
selector candidates. Direct-head fitting is supervised, with saved optimizer
and sampler/global RNG states; it is not PPO.

Conservatively charging all four GPUs from job creation through scheduler
closure gives 29,008 GPU-seconds for this attempt, plus 2,924 and 720 for the
two failed attempts: **32,652 GPU-seconds = 9.07 GPU-hours**, below the approved
12-hour ceiling. Setup, failures and full allocation waiting are included.
Pre-existing S8 teacher histories are inherited data, not new collection.
[Terminal accounting](TERMINAL-ACCOUNTING.json). No new allocation or PPO run
followed this screen.
