# Gate readiness, 2026-11-02

Human-directed scope: gate0 is Flora agreement/headroom; gate1 is simulator-
and Flora-reward positive controls. The active RLinf action-expert learner is
the simulator-reward half only. Gate4 needs a frozen-motor baseline, improving
multi-task histories and a frozen history controller on the SAME interface.
This work adds CPU/offline analysis and implementation; no competing GPU PPO
job, no edits to active worktrees, no new robot histories.

1. Recompute terminal threshold/P5 sweeps on all634 saved scores, including
   successes. Split384 TRAIN episodes by native case, select on fit only, and
   report internal validation and250 calibration episodes separately. Candidate
   screen: precision lower95% >=.95, FPR upper95% <=.05, recall >=.50. These
   are development screens, not an independently passed production gate.
   Terminal scores cannot measure temporal holds or first success. Never use
   a whole-video score as an earlier prefix decision.
2. Decompose four completed native8 banks into per-case k/8, all-fail/all-
   success/mixed/3–5-of8, pair disagreement and within/between-case variance.
   Reuse audited evidence only. These are selected native failures at H600;
   eight draws do not establish irreducibility or attainable selection gains.
3. Build a common frozen-motor PPO port: public prefix/proprio features before
   action sampling; native noise and steering both inside the policy action;
   explicit deterministic action map shared with the meta-learner; first-
   success returns, gamma.99/lambda.95; practical optimizer batches; full
   Adam/sampler/environment/Python/NumPy/Torch/private-generator recovery.
   For noise32 reproduce the gate0 tied-tanh map. For other interfaces retain
   native iid50x32 sampling, now in the scored joint action. This changes
   action dimension, so the selected interface needs real native numerical
   and simulator qualification before launch. It is not full-weight RLinf.
4. Common prospective evaluation:200 independent canonical cases/task,
   explicit draws per case, full/memory-reset/reward-blind interventions,
   identical attempt-specific streams, first-success and final success both
   saved. Fixed physical horizons prevent success termination revealing truth
   to reward-blind evaluation. Bootstrap and exact tests cluster draws by case.
   The adapter must verify native state/cameras; a declared seed is insufficient.
   Existing H400/H600/H750/H900 panels remain separate.
5. Gate4 fit profile:plain causal decoder,128 context AND128 training window;
   balanced histories from OpenDrawer, PickPlaceDrawerToCounter, CoffeeSetupMug,
   TurnOffStove only after each source learner improves. Keep whole source
   streams disjoint across train/validation, with leave-one-task-out folds for
   task generalization. Four tasks give a limited generalization test; never
   describe all four as unseen if all four enter training. Keep flat existing
   histories as pipeline checks. Qualify the AD-trend language in both docs.

## Temporal reward experiment prepared by the offline result

For the saved videos, first read native success_steps/first_success_step
sidecars without replaying controls. Score causal prefixes ending at50,100,...
(and finer saved-frame endpoints around native first success) under a NEW
cache/protocol. Include native never-successes, early incomplete prefixes and
successes. Do not crop relative to native success for policy evaluation:
truth-centered crops are an analysis-only diagnostic. Compare holds1/2/3 at
the actual stride; a hold over chunk decisions is not a consecutive RGB-frame
test. Record confirmation time, never backdate it. Fit on development cases,
freeze a rule, qualify on independent cases at the intended horizon/cameras,
and report precision, recall, early errors and signed timing in controls.

No temporal scores exist in the saved terminal labels. Prefix rescoring needs
Flora inference and its own bounded execution binding; do not report fabricated
timing metrics or claim that tightening the terminal threshold fixed reward.

## CPU validation

Use the existing exclusive Baige allocation, hide CUDA, cap CPU threads at2,
run tests in a NEW detached checkout of this commit. No model inference or
simulation runs in the active PPO process. Run OFFLINE.py against retained
data, save checks and readouts here, and record actual Linux host/source.
Deploy code through Git; output JSON is an artifact. This is not authorization
to use DeltaAI or to expand a scientific GPU budget.
