# Concrete first allocation

Four instances, one GPU/32 CPU/128 GiB RAM/16 GiB shared memory each, at most
180 minutes, aggregate ceiling43,200 GPU-seconds. One task per instance and
eight native lanes per GPU. Source and dependencies are immutable Git pins;
all output goes to this run's separate shared project directory. No scorer
process and no PPO optimizer. Direct-head distillation is the only new fit.

Each task identifies the first20 failures among up to104 fresh TRAIN anchor
cases at600 controls. Screening native seeds start at211100 (OpenDrawer),
231100 (DrawerToCounter),251100 (CoffeeSetupMug),241100 (TurnOffStove).
Selection uses an independent anchor draw. All anchors are charged. Cases200
through207 in the same prospective bank are reserved for direct-head
competence qualification; DEV/ID/OOD banks are untouched.

Continuous arms use160 episodes/task: native resampling, action0.25,
action0.5, bounded32D tied initial noise, final-layer KV-value residual, and
velocity residual. A qualified direct head gets the same160-episode bank.
Each native case has eight draws and twelve chunks per draw. Motor schedules
match across applicable arms; actor draws match across action bounds. Pure
tied noise replaces native noise and is reported as a different distribution.

The conditioning projection is a fixed Rademacher32D map (seed921003), added
to final-layer cache values, scaled by their per-example RMS. Candidate
fractions0.01/0.05/0.25 are evaluated on eight public anchor observations.
Velocity correction is a repeated32D tanh code with fractions0.05/0.25/1.0
at every native Euler step. Pick each family's one fraction closest in decoded
normalized action RMS to the action0.25 perturbation; no success labels enter
that choice. The actual-model zero-control replay must be bitwise identical
to the native draw, and nonzero interventions must be finite and nonzero.

The direct head2064→256→600 uses public state and frozen prefix features;
all24 base-action features are removed. Motor-action labels come from retained
S8 pending records plus new native anchors, with every fifth complete episode
held out. At most4000 AdamW steps, validation every50, plateau after12 checks;
keep full latest optimizer/sampler/RNG and selected weights. Require validation
and task-anchor MSE≤0.25 in norm-std units, then an eight-case TRAIN
competence pilot (at least one success and no more than one success below its
paired native panel). This small preliminary gate does not establish motor
equivalence. A failed gate leaves this family incomplete, rather than calling
an incompetent head evidence against direct-action PPO. Its screening arm
uses the same6D stochastic correction as action0.25 around the direct head;
the direct head supplies all50×12 base actions without using expert actions
as inputs. Current implementation still computes unused expert outputs for
the common feature capture; that inference cost is charged.

The realizable selector gets one rollout per failure case, eight native
candidates per chunk. A frozen saved simulator-PPO value network scores each
candidate's public state/prefix/base-action features. This is a disclosed
heuristic, not an oracle or a qualified Q-function; it may be weak off the
OpenDrawer source distribution. Save every candidate motor seed, selected
index and actual selected seed. Compare its one-attempt outcome with native
draw0; report native best-of-eight separately. Candidate inference is charged
explicitly. No new selector fitting or deployment simulator access is used.

Each reset must match exact physical/model/camera/public state to a shared
canonical snapshot. Keep original policy images and permit only the already
qualified one-level uint8 render roundoff. Save initial snapshots, complete
three-view videos, per-chunk motor features/base/executed actions/controls and
independent simulator truth. The existing AD/DPT and600-calibration jobs are
not touched. Master exit waits for all four task nodes, and its full wait is
included in allocation accounting.
