# Targeted comparisons before full collection

The active user goal asks for DSRL versus meta-learned noise steering on LIBERO
and RoboTwin, targeted subsets and several scales before expanding collection.
This run advances the LIBERO diagnostic. RoboTwin retains its separate owner;
the larger32-GPU-hour LIBERO allocation request remains pending.

Source data: one qualified Spatial00 history,20 whole cases/80episodes/
742actual chunks,7,892 source actions,4,920 SAC updates. Source-local NLL
validation: disjoint states40–49,617chunks,6,538actions, independent SAC
teacher. All source/teacher costs remain counted once. No LIBERO90, Plus or
Pro data enters new controller pretraining.

Optimization axis: four AD/DPT full/static controllers, same width128,
two layers/four heads/context300, batch8/window128, LR3e-4, seed0. Fixed
exports at25/100/500 updates along one optimizer trajectory per controller;
500 final updates each, not625. Exported full recoveries retain optimizer,
sampler, Python/NumPy/Torch/CUDA RNG. Validation NLL does not select the
robot-test weights. The initial5-minute/controller limit was undersized;
the uniquely named corrected run allows10minutes/controller and45minutes
overall. Its rerun is not labeled a continuation of the initial optimizer.

Data-amount axis is separate: first8/16/20 whole chronological source cases
contain318/603/742chunks. The8-case prefix has no source-policy update and
is BC/DPT-only, inadmissible as AD. The16/20 prefixes expose real source
learning clocks. Every DPT scale retains the final20-case teacher and its
complete generation cost; this is not a DPT source-interaction saving.

Primary robot diagnostic: predeclared LIBERO90 tasks0 and1, state0, seed123,
with native frozen motor, frozen same-corpus source-SAC and AD full/static
at25/100/500 updates. Each method/case receives an independent zero-support
query and one actual support episode followed by a separate query, with
paired case/role randomness. Eight methods×two cases×three episodes=48
episodes, worst-case19,680 actions including settling. No deployment SGD.
Optional AD500 episode-reset and tied-Gaussian groups run only after every
primary group, in fixed order, if a full worst-case group reserve fits.
DPT models are preserved but DPT robotic comparison is deferred in this slice.

The diagnostic escrows20,000 actions from the campaign's21,056 previously
unassigned actions through the canonical `general-ledger.json`; the earlier
failed typed-JAX qualification's10 settling actions remain charged, leaving
1,046 unassigned. The original SAC/PPO reservations and1M benchmark total
are unchanged. Target costs are reported separately and target traces remain
evaluation-only. The original campaign's source-only data contract remains
unchanged; this is an explicitly accounted diagnostic allocation.

Evaluation retains raw current/next camera/proprioceptive observations,
actual noises/actions/rewards/boundaries, support/query roles, state hashes,
fixed policy hashes and complete episode-boundary context/RNG recovery.
RPC intents are journaled before execution and unresolved calls count at
their full ceiling. Resume must retain source, run, cap and checkpoint hashes.

Source SAC uses the authors' pixel64/proprio8 encoder. Students use frozen
VLA-prefix4104 features. Their robot difference does not isolate memory;
AD full versus trained-static and episode-reset comparisons address context.
Frozen source-SAC is not ordinary target-online DSRL; that required comparison
remains outstanding. Single-goal source data and two target cases establish
neither broad transfer nor a negative verdict on the meta-learning method.
