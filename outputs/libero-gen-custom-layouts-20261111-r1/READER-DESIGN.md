# Reader experiment after physical admission

This is the next-stage design, not a training launch or a result. The current
4 GPU-hour allocation covers custom-layout qualification only. Freeze the
collection counts, reader configuration, seeds and incremental compute cap
after geometry and motor admission pass, before collecting training data.

## Question and limits

Can a learned reader use previous native outcomes to select the correct
frozen motor prompt in a fresh scene, with all weights fixed at evaluation?
The hidden variable is LEFTMOST versus UPPERMOST in the fixed agent-view
image. Both rules appear in training. This is binary task identification
with motor noise, not unseen-rule generalization or learned exploration.
With perfect execution, a single informative disagreement attempt can
identify the rule. A rising curve over several attempts would not by itself
establish a general evidence-accumulation algorithm.

The selected candidate must vary with both rule and query scene. Within
each stratum, each rule selects candidate 0 and candidate 1 equally often.
Maintain this balance in each data split, not just the development panel.
The 16 admission layouts are development evidence and cannot become the
reader's final confirmation set. Fresh jitter within the qualified layout
families is new scene sampling, not a claim of broad visual generalization.

## Data and leakage controls

Each session begins with an agreement demonstration, followed by a fixed
budget of disagreement attempts and a fresh disagreement query. Choose
attempt candidates with a schedule independent of the hidden rule and all
outcomes. The agreement demonstration has the same correct candidate under
both rules, so it cannot identify the rule by itself.

Collect native physical executions of both candidates on the sampled scene
bank. Reuse those executions to form paired hidden-rule sessions; the rule
changes only the native outcome label assigned to an otherwise identical
execution. Sample whole physical scene/trajectory groups into train,
validation or confirmation before composing sessions. Counterfactual rule
copies, both candidate executions, reused frames and all derived feature
caches from one base scene stay in the same split. Report physical scene
and rollout counts separately from the much larger number of composed
sessions. Reusing a finite scene bank does not create independent evidence.

Follow PLAN.md's selected-skill stopping and fixed-horizon recording
contract. Use the same public bytes and candidate schedule in true,
masked and shuffled conditions. Hidden-goal values cannot affect trajectory
length, padding, filenames, sample ordering or candidate choices. Public
inputs may contain RGB, wrist RGB, proprioception, the executed candidate
and the assigned feedback token. Neither privileged positions, physical
object IDs, task paths, rule labels nor both native predicate values enter
the learner. Keep privileged fields only in a separate audit record.

Execution failure remains in the dataset. It is not automatically evidence
that the selected candidate was semantically wrong. Record both official
native predicates and selected-skill success for diagnosis, without giving
those extra labels to the primary reader. Do not relabel a failed selected
skill as a successful alternative.

## Qualification before a reader fit

1. Pass geometry/visual admission and the preregistered 32-episode motor
   panel, retaining all failures and repeated smoke identifiers.
2. Audit public-record equality across counterfactual hidden rules. The
   allowed difference is the explicit outcome channel. Verify split groups
   from the actual manifest and cached-feature ancestry.
3. Qualify the visual interface with a rule-given positive control on fresh
   development layouts. If the chosen frozen visual features cannot support
   selecting a motor from the image and supplied rule, diagnose perception
   or the candidate interface before interpreting a history-reader failure.
   The rule-given model is a separately labeled control, not a valid input
   to the primary learner.

Use a small offline-trained Transformer to select one of the two frozen
motor prompts. Freeze its architecture, objective, history budget and
selection rule before fitting; preserve resumable optimizer, sampler and
RNG state. Any visual feature extractor and the BPP motor stay frozen.
If features come from the released BPP checkpoint, record the exact layer
and preprocessing and qualify the extraction path on the allocated GPU.
The reader and motor are both frozen for confirmation. No recurrent-state
mechanism claim follows merely from feeding the complete history again.

## Paired readout

The primary contrast is the same frozen checkpoint on identical histories
and queries with true feedback versus masked feedback. Also include:

- Demo-only and cleared-history inputs at the same query scene.
- Shuffled feedback with the positive marginal preserved; report the
  fraction of tokens actually changed and the shuffle unit. Shuffling and
  masking can introduce distribution shift, so neither alone is sufficient.
- A matched donor-history control, with donor selection fixed in advance.
  Distinguish same-rule donors from opposite-rule donors rather than mixing
  them into an uninterpretable average.
- A query-image intervention, to check that the reader uses the fresh scene
  instead of predicting a fixed prompt once it identifies the rule.
- The rule-given visual positive control and the empirical motor ceiling.

For exactly balanced disagreement queries, a rule-blind semantic choice is
50% in expectation; physical native success can be lower because the motor
can fail. Report semantic candidate accuracy and native physical success
separately. Selecting from an already collected pair of executions gives
an offline replay success estimate. Label it as such; a closed-loop robot
claim requires fresh execution of the reader's selected candidate.

Report per-fit results and paired uncertainty over independent base scenes
or sessions, accounting for reuse of physical trajectories. Do not treat
every recomposed history as independent. Fit several fixed seeds, use only
validation for selection, and run the locked confirmation once. A
same-history reference cannot use privileged geometry or both hidden native
outcomes without being explicitly labeled a privileged oracle; do not
compare such a reference as if it had the reader's information.

A positive outcome supports useful feedback-conditioned prompt selection
on this two-rule task. It does not establish a particular internal belief
state, Bayesian updating, exploratory action selection, held-out rules,
continuous-action adaptation or the separate pi0.5 history interface.
