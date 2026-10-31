# Frozen fitted frontend plus physical4D

This prospective hybrid interface preserves the six source learners and their
4D action coordinates. It does not fine-tune a complete fitted head, warmstart
SAC from that head, or rehabilitate the old fixed-map negative A1 result.

The exact order is native prefix/sampling/decoding with one frozen fitted KV
or wide-action frontend, then the reviewed fixed4x14 arm-only physical map,
then public requested-command recording, then the existing latent calibration,
then native execution. The physical map retains its original basis, temporal
functions, coefficient bounds and small arm-joint bound. It clips corrections
to remaining12-arm URDF headroom. It leaves raw gripper requests unchanged.
The fitted wide frontend itself retains2rad/1.1 bounds; the extra physical
stage is separately bounded and is part of a new composed mapping.

Zero physical coefficients reproduce the fitted frontend's command bits,
including raw gripper overshoot and signed zero. A fitted baseline with arms
outside the reviewed limits is rejected rather than repaired. Explicit
`arm=native`, coefficients0 disables BOTH frontend and physical hooks for the
raw-native comparator. The source mapping uses `arm=action` for either fitted
frontend kind. Nonzero physical coefficients never imply raw-native parity.

The optional backend `install_post_decode_residual` API installs one eval hook
before any raw observation or corpus registration. Its defaultNone path adds
no output contract/metadata and preserves existing behavior. It does not use
the calibration-transform slot. Metadata preserves frontend output, physical
requested/applied deltas, pre-condition public14 requests and audit-only actual
conditioned commands separately. Public student2076 observations remain
2062 native features plus the previous last-valid pre-condition14 request,
zeroed on every physical reset. Source/AD labels remain actual bounded4D
requests. DPT still requires a physically qualified same-mapping TRAIN1 actor
and replaces only query targets. Full14 commands are never coefficient labels.

## Shared explicit entry

`rl.robotwin_fitted_physical.create_control` is the one factory used by:

```sh
python -m fluffy_bassoon.meta_rl.robotwin_fitted_source source <original bound source arguments>
python -m fluffy_bassoon.meta_rl.robotwin_fitted_source readout <original bound readout arguments>
```

The future source-parallel/readout APIs accept an optional `control_factory`;
defaults still select the original `CoefficientSteering`. Source also accepts
an optional selection validator, used explicitly by this new entry. Spawned
workers receive the actual importable factory, without global monkeypatching.
Legacy serial and fixed mapping implementations are unchanged. A default old
factory cannot match this hybrid mapping's SHA and fails before a reset.

Copy the reviewed physical interface to a NEW hash-bound interface file and
add `fitted_frontend` with format `frozen_fitted_frontend_physical4d_v1`:

| Binding | Meaning |
| --- | --- |
| `kind`, `iteration`, `checkpoint:{path,sha256}` | Existing fit ABI; kind kv/action; explicit final5000 endpoint |
| `fit_source_commit`, `fit_contract_sha256` | Exact producer and complete endpoint contract |
| `fit_profile_format`, `fit_profile_sha256`, `fit_complete_cache_sha256` | Exact declared all12/8TRAIN4VAL fixed-final fit and cache profile |
| `fit_provenance` | COMPLETE endpoint contract verbatim, including diagnostic purpose, actual outcomes and source-plan/cache references; never assume all12 succeeded |
| `head_config_sha256`, `head_state_sha256` | Canonical JSON configuration hash and existing parameter_digest of all state tensors |
| `motor_sha256`, `normalization_sha256`, `joint_convention_sha256`, `base_interface_sha256` | Unchanged motor/norm and reviewed source conventions |
| `physical_basis_sha256` | Existing fixed-basis digest, verified against the constructed4x14 basis |
| `wrapper_sha256`, `backend_sha256`, `fixed_control_sha256` | Exact executing mapping/backend/fixed-map source files |
| `frontend_implementation_sha256`, `residual_primitives_sha256` | Exact wide-or-KV implementation and shared residual primitives |

The loader uses the source owner's existing `{kind,iteration,contract,config,
policy,diagnostics}` endpoint ABI. Wide heads reconstruct through
`WideActionResidual.from_config` and its protected strict state load. KV heads
reconstruct their declared2062/public, rank4, last-layer, KVH1/HD256, scale0.1
configuration. Both enter eval mode with all parameters frozen; every control
step checks their original state digest. No learner/serializer is introduced.
Original positive-bank producers retain their original positive-seal gates;
an explicitly declared all12 command-capacity diagnostic retains failed04,
all12 identities, the8TRAIN/4VAL split and exact diagnostic cache/provenance.
Neither its endpoint nor its command loss implies original primary-bank
success. A later actual composed-interface qualification is still mandatory.

Both source and student profiles must bind this NEW control-interface SHA and
composed mapping SHA. The immutable endpoint reference must remain accessible
through source collection, teacher queries and readout. The qualification
reference is hash-bound through the new selection/profile rather than included
inside its own mapping hash, avoiding a circular mapping/proof checksum.
For existing prefix admission, use explicit format
`fitted_physical_control_qualification_binding_v1` with original immutable
references named `control_interface`, `source_profile`, `selection`,
`driver_plan`, `driver_qualification`, `a1_result` and `completed_episode`.
Only this format dispatches to the new composed proof verifier; the old native
qualification branch is unchanged. The source profile is the original raw
SourceProfile JSON and binds the exact selection file SHA. Driver plan binds
the interface/scope, while selection binds its rule/A1/qualification refs.
The subsequent existing A2/improvement/TRAIN1 gates still run independently.

## Qualification and caller-owned resources

There is no source/data admission yet. The new selection schema is
`fitted_physical_source_selection_v1`; old A1 receipts are rejected. A new
`fitted_physical_motor_qualification_v1` must prove zero-to-fitted identity,
separate raw-native identity, unchanged frozen frontend and motor, bounded
physical intervention, correct public-request/calibration order and a complete
actual physical episode. A1 format `fitted_physical_A1_v1` supplies complete
raw-native development records and two independent paired confirmations of
this composed mapping:2/2 rescue versus0/2 raw-native, with identical actual
initial input/instruction and paired motor draws. Canonical manifest/chunk
hashes, natural terminal boundaries, actual steps and success are checked.
Qualification boolean fields are `fitted_baseline_zero_exact`,
`raw_native_control_exact`, `physical_delta_bound_passed`,
`physical_nonzero_effect`, and `public_request_before_calibration`; frozen
frontend/motor identity and zero gradients are checked independently.
The rule format is `fitted_physical_A1_rule_v1`, confirmation_repeats2,
comparator raw_native, baseline frozen_fitted_frontend, candidate
frozen_fitted_frontend_plus_physical4d. The new gate never uses A2 outcomes
to select the natural A1 winner.

The SAME mapping, physical basis and existing calibration pair must then pass
the original A2 cross/teacher/improvement gates. No old inverse qualification
is inherited through clipping or a different fitted baseline. A new exact-code
CPU qualification, actual allocated zero/raw/frozen tests and concurrency
capacity proof are required. Existing action budgets and resource tickets
remain the caller's responsibility. This factory hardcodes no job, GPU cost,
journal tip, allocation or renewed phase allowance; a later explicit linked
resource continuation must be admitted by the sole runtime owner.

Prepared tests, ONLY on allocated Linux CPU:

```sh
uv run pytest -q tests/test_robotwin_fitted_physical.py
```

They cover both endpoint kinds, frozen-hook preservation, zero-to-fitted bit
identity, physical bounds/headroom, grippers, canonical intent versus actual
conditioned commands, public2076 reset/history, source/readout factory/mapping
identity, provenance failure and rejection of old A1. Synthetic fixtures admit
no fitted frontend, source history, AD/DPT dataset or runtime job.
Broader unchanged suites are not part of this focused request. The runtime
owner can qualify default/factory integration separately under its next
explicit CPU continuation if needed.
