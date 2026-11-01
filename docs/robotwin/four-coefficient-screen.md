# Bounded frozen-Motus coefficient screen

Executable: `python scripts/diagnostics/robotwin_steering_screen.py`.
This is the first-stage implementation of the shared targeted-pilot scope,
not evidence of live qualification or steering benefit. Existing DSRL/native
jobs, old source campaigns, default samplers and NoiseSAC defaults are untouched.
The sole existing resource owner submits/deploys; this driver schedules nothing.

## Exact control API

`rl/robotwin_steering.py` exports `CoefficientSteering`, `FixedKVResidual`,
`FixedActionResidual`, `clipped_coefficients`, `validate_interface`.
The wrapper uses the existing scalar `RobotwinNoiseEnv`: frozen Motus through
pinned RLinf, H32/full32D native noise, five flow passes, three RGB views,
2062 public features and14 decoded ALOHA commands. No alternative motor loads.
Actual model/norm files, source revisions, cache shapes and action conventions
are validated before control; `task_config/_eval_step_limit.yml` supplies each
task's horizon without a400 fallback.

Every arm accepts four float32 coefficients, clipped to[-1,1]. Fixed bases use
NumPy PCG seed20261031 (noise),20261032 (KV),20261033 (action); bases themselves
are preserved in recovery and bound by typed-array hashes. The temporal basis
is `[1,2x-1,sin(pi*x),cos(pi*x)]` over normalized native action slots or valid
prefix-token rank. Coefficients remain constant within an A1 episode.

- Noise: independent full H32×32 native Gaussian plus a fixed-basis delta
  `noise_delta_bound*sum(c_r*temporal_r*basis_r)/4`. The delta is bounded;
  total native Gaussian noise is deliberately **not clipped**, preserving
  native behavior at zero coefficients.
- KV: the same linear `/4` rule edits only the final native layer's post-RoPE
  K/V, with declared cache-unit bound, mask, dtype, layout and padding intact.
  This is a fixed rank<=4 search family, not a trained recurrent KV bridge.
- Physical: `arm_joint_bound*tanh((temporal*c)@basis)` in decoded arm radians,
  **without `/4`**. Grippers6/13 are copied exactly. The seeded constant basis
  row has support only on joints0/1/7/8; other rows are seeded arm directions.
  There is no guessed absolute-command clipping or implicit joint conversion.
  The first draft's linear physical `/4` mapping was replaced before any pin,
  test allocation or live use; its narrower reach could not support the
  proposed controlled calibration. Mapping hashes bind the final choice.

`physical_basis()` returns a copy of the exact4×14 physical basis.
An aligned calibration `b=arm_bound*tanh(theta*basis[0])` has the candidate
odd inverse `c=[-theta,0,0,0]`, |theta|<=1. This algebra does not prove actual
actuator parity, range headroom, clipping behavior, or an inverse for KV/noise.

Construct `CoefficientSteering(env, interface_contract, interface_sha256=...,
post_command_transform=None, condition_contract=None)`, then `attach_corpus`
before the first reset. `set_coefficients(arm, c)`, `reset(task, seed,
motor_seed=..., paired_instruction=None)`, `step(c)`, `snapshot()`, `physical_basis()` and
`mapping_sha256()` form the public API. `post_command_transform` and
`condition_contract` are public bindings. Optional calibration is called as
`transform(base_decoded, requested_after_mapping) -> (final_commands, audit)`;
shape/dtype/device/finiteness and exact grippers are checked. Agent history is
the **pre-calibration** request; final actuator commands and latent details
stay under `audit_only_post_command_transform`, never in public features.

## Complete episodes, raw corpus and A2 integration

`meta_rl/robotwin_steering_screen.py:run_complete_episode` is the common loop.
It accepts a control, dict spec, canonical CampaignBudget, run_id and phase.
Spec fields: id/task/case/reset_seed/motor_seed/arm/coefficients/horizon;
optional method, policy_version, expected_reset, paired_instruction. `method` is preserved for
lineage accounting; it is not replaced by the mapping arm. A whole native
episode must fit before reset; each chunk receives a durable intent before
execution. No episode is cut to meet a quota. Reset-policy actions must be
source-verified zero; nonzero/unmeasured settling controls are rejected rather
than omitted. Internal non-policy setup physics remains explicitly separate.

Optional controller methods: `act(public2062)`, `decision_metadata()`, and
`observe(obs, actual_coeff4, reward, next_obs, terminated, truncated,
primitive_steps, corpus_ref)`. Observe runs only after a valid committed chunk
and accounting resolution. Learner updates belong to the caller after the
complete receipt, never this loop. `on_chunk`, `on_episode`, and `before_step`
callbacks do not add a simulator/logger/ledger. Receipts bind complete raw
episode manifests plus chunk SHA refs, actual costs/reset content and control
RNG. Partial episodes are never exported as complete.

For every repeated scene, use `paired_reset_fields(original_complete_record)`.
It returns `expected_reset` (actual seed/raw-observation/public-feature hashes)
and `paired_instruction` (the original one-element actual instruction list),
checking that the baseline receipt and reset identity agree. The common runner
requires both fields when an expected identity is supplied. Source-validation
milestones, condition-cross and student support/query or lesion pairs share
this API. No stage rewrites text independently.

The backend binds this instruction before every encoding and raw corpus frame
for the whole episode. Simulator-sampled text remains separately in
`reset_info.instruction_binding`, decision metadata `paired_instruction` and
each next native result's `info.paired_instruction`; audit never enters public
features. Raw RGB/proprioception are untouched and all identity guards remain
exact. An unpaired reset clears the binding. The initial paired qualification
also repeats the actual effective input and requires exact public features,
prefix mask and native K/V cache before decoded zero-parity/frozen checks and
before any real command. This numerical check adds no robot episode.

Typed decision metadata contains authoritative `steering_coefficients` (the
actual clipped4D choice), separate `proposed_steering_coefficients`,
`steering_mapping_sha256`, `steering_interface_sha256`; caller-provided audit
fields are `steering_lineage_id`, `steering_condition_id`,
`steering_partition`, `optimizer_updates`, `source_warmup` and actual
`policy_version`. All audit IDs/conditions stay outside model inputs. Existing
native sampled_noise remains the real native noise, not a KV/action label.
`student_previous_agent_command` is the causal14D previous agent-requested
command, selected using the LAST TRUE native execution-mask index (holes are
not treated as dense execution). It is zeroed on EVERY physical reset. The
wrapper's `student_observation(native2062)` returns2076 features for downstream
full/static students; the teacher/native API remains2062. Exporters concatenate
the native current features and this exact typed field; online evaluation uses
the same method. DPT labels pass only the native2062 slice to their teacher.
Neither unexecuted end-of-chunk proposals nor post-calibration actuator commands
enter this field. Full neural memory may retain support tokens; static/current
query inputs do not inherit the preceding support's command after reset.

`normalize_arm` maps the three long semantic descriptions to canonical
`noise|kv|action` at the interface boundary; mapping SHA binds the short key.
Profiles must use that same canonical selected-arm identity. Tests cover alias
identity as well as complete episode/manifest and typed command accounting.

## Executable A1 plan and boundaries

Qualification uses the first planned real episode; no extra robot episode is
hidden. At identical actual observation/noise it checks native cached parity,
zero learned-KV, zero fixed-KV and zero action decoded commands, native-cache
nonmutation, a nonzero actual frozen-suffix gradient into adapter output bases
with no motor gradients, and unchanged full motor hash. One complete real
episode is required before the receipt says passed.

Eight predeclared source dev resets each on phone/object stand are run first.
Choose the first task in that fixed order with a native failure; at most the
first three failed cases are searched. Three seeded uniform coefficient
candidates per arm/case are ranked by success then completion cost. A successful
winner gets two independent motor-noise confirmations with two matching native
controls per case; failed search families get no invented success/retry.
All32 locked query cases remain unopened.

For verified400/400 horizons, the maximum is
`8*(400+400)+3*[3arms*(3search+2confirm)+2native]*400 =26800` actions,
67 complete episodes. This includes per-arm confirmations, paired native
controls and the qualification episode. The executable computes the bound
from actual native horizons **before outcomes**; if horizons grow it records a
smaller case count (at most3) or rejects the plan. It never changes the grid
after observing outcomes or cuts episodes. The unused3200 is not extra search.
Search failure is inconclusive beyond this restricted class; selected maxima
are biased development results, not held-out performance, RL or ICRL evidence.

## Admission and CLI handoff

The caller supplies SHA-bound SCOPE, current allocation ticket, interface and
joint-conventions receipts. SCOPE binds ONE28800GPU-s/300000actions/64GiB phase
and first-stage7200GPU-s/30000actions; all stages/failures/retries count.
The allocation ticket binds source_commit/hostname/GPU1/output, scope_sha256,
phase_output_root, original driver_deadline_unix, reviewed initial_episode_seconds,
reserved_gpu_seconds/prior_phase_gpu_seconds/other_reserved_phase_gpu_seconds/
prior_a1_gpu_seconds and the sole canonical action_ledger.
No clock is reset or resource granted by this driver.

Canonical phase caps are a1=30000,source=120000,source_validation=30000,
condition_cross=15000,student_evaluation=105000. Exact six source_caps keys are
`sac_calibration_{minus|plus}_{train0|train1|validation2}`; caller supplies each
numeric cap, at most48 verified native episodes, subordinate to shared caps.
Ledger contract binds phase_id, scope_sha256, motor_sha256 and candidate_tasks.
No A1-only header, child import, closed-grant borrowing or second ledger exists.

The interface includes all validated dimensions, basis_seed, noise_delta_bound,
kv_delta_bound, arm_joint_bound, motor_sha256, normalization_sha256,
joint_convention_sha256, finite joint_lower/upper[14], radians and absolute
gripper conventions. Convention receipt binds exact source commits/files,
reset_policy_action_bound0, reviewed maximum_arm_delta and12 joint_limit_records
(root/path/SHA/joint name/action_index). The driver parses each source URDF's
unique limited revolute joint and verifies declared arm ranges. Arm correction
cannot exceed one tenth of any verified arm range. This is source binding,
not a claim that every native proposal has actuator headroom.

```sh
python scripts/diagnostics/robotwin_steering_screen.py \
  --scope SCOPE.json --scope-sha256 SCOPE_SHA \
  --allocation ALLOCATION.json --allocation-sha256 ALLOCATION_SHA \
  --interface INTERFACE.json --interface-sha256 INTERFACE_SHA \
  --joint-conventions JOINTS.json --joint-conventions-sha256 JOINTS_SHA \
  --source-commit EXACT40HEX --run-id FRESH_RUN --output FRESH_OUTPUT \
  --model-path EXACT_MOTUS_ROOT --norm-stats EXACT_NORM_FILE \
  --rlinf-root EXACT_RLINF_ROOT --robotwin-root EXACT_ROBOTWIN_ROOT \
  --gpu-probe ACTUAL_ALLOCATED_GPU_PROBE.json
```

The owner supplies the qualified environment/renderer configuration and outer
allocation timer; this CLI does not change GPU visibility or dispatch jobs.
Recovery saves exact bases, search plan/cursor, private motor-noise generator,
global Python/NumPy/Torch/CUDA RNG, raw roots and canonical action journal.
Only same-code closed operational boundaries resume to a fresh output using
`--resume` plus exact SHA. Nonzero/unknown-charge/partial episodes block
automatic continuation. Terminal scheduler GPU costs are settled by the
resource owner against the same cumulative reservation, separately from driver
wall time. Fresh CPU qualification covers the new mapping/driver/backend tests;
the earlier41-test hook result does not qualify this new executable.

An explicit diagnosed instruction-mismatch migration is the sole exception to
the same-code boundary rule: `--resume ... --resume-sha256 ...` plus
`--paired-reset-migration ... --paired-reset-migration-sha256 ...`. It accepts
only the exact pre-command paired-reset RuntimeError after the complete native
bank, binds predecessor result/recovery/terminal/cost/closeout and actual raw
reset/NPZ/zero-chunk manifest evidence, checks original episode language
constancy, and retries only the identical pending search with original text.
Old records and outputs remain immutable. Under the canonical ledger lock the
old tip and all actual charges must remain, with zero pending/uncertain costs
and no command reservation for that failed case; only this fresh run's source
registration may follow the old tip. It grants no new resources, repetitions,
refunds, or general failure-resume switch. The source owner supplies and reviews
the SHA-bound proof after predecessor termination.
