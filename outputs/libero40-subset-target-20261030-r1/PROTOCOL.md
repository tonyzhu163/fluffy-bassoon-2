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
with native frozen motor, tied Gaussian noise, frozen same-corpus source-SAC,
AD full/static at25/100/500, DPT full/static at500, and AD500/DPT500
episode-reset lesions. Each method/case receives an independent zero-support
query and one actual support episode followed by a separate query, with
paired case/role randomness. Thirteen methods×two cases×three episodes give
78 primary episodes, worst-case31,980 actions including settling. No
deployment SGD. Both AD and DPT receive direct fixed-weight memory lesions.
DPT25/100 full/static groups run only after all primary groups, in fixed
order, if a full worst-case group reserve fits.

The diagnostic escrows33,046 actions:21,046 from the general reserve through
`general-ledger.json`, with the failed typed-JAX qualification's10 settling
actions retained, plus12,000 explicitly marked evaluation-only in the PPO
pool. The future full/reset PPO reservations drop equally from200k to197k;
PPO pool allocation becomes19,500 original qualification +394,000 newsource
training +64,000 sourcevalidation +12,000 targetdiagnostic +500 contingency
=490,000. No active/completed reservation or1M benchmark total changes.
Target costs are reported separately and target traces remain
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
