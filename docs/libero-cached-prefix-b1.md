# Cached-prefix scalar motor adapter

`vla_cached_scalar_motor.CachedScalarMotor` is an opt-in wrapper for the native
`FrozenOpenPI` pi05_libero motor. It reuses the scalar prefix/suffix equations
qualified atc6f46e8, keeping B1 model kernels. It changes neither the controller,
simulator order, H10/32D noise, trial RNG, PPO batches nor SAC update schedule.
It is a candidate until its exact code passes an allocated runtime comparison.

The native policy object stays accessible as `motor.policy`, including
`policy._model` for frozen-parameter hashes and `policy._rng` for existing
complete-boundary recovery. Metadata, native horizon and the original feature
contract remain unchanged. `execution_metadata` separately records cache use
and qualification limits; a native feature-space contract is not a claim that
cached execution has been qualified.

The feature callable prepares a B1 prefix for the actual complete observation
content and instruction. Repeated feature reads return copies. A matching
motor inference consumes that KV exactly once and advances the native motor
RNG once. There is no new sampling stream, noise clipping or rescaling: finite
noise shaped[10,32] is retained, including unbounded SAC warmup noise. A present
stale cache fails before sampling. With no pending prefix, native `infer`
remains available, including the original initial same-noise checks. A failed
suffix attempt retains its one native RNG advance and cannot retry or reuse KV.

The original `feature_encoder.contract`, `.calls`, `.cache_hits`,
`.elapsed_seconds`, `.qualify` and `.diagnose` interfaces remain available.
`qualify`/`diagnose` preserve the original native feature checks and clear
ephemeral state. They do not substitute for native-versus-cached action and
feature qualification. Model weights are assumed immutable between the
feature query and inference; full frozen-weight hashes belong to the actual
runtime qualification, rather than a host transfer of all weights per query.

## Integration after qualification

Production entry points remain unchanged by this candidate. Both source
batch14 and source SAC can opt in at their existing single-motor construction
boundary after the owner binds a passed exact-code runtime receipt:

```python
native = FrozenOpenPI(args)
motor = CachedScalarMotor(native)
write_json(attempt / "motor-execution.json", motor.execution_metadata)
```

Keep the original scientific motor/feature contract, and retain the separate
execution receipt with its qualified implementation/input/runtime hashes.
Do not silently migrate an active recovery or change a canonical ledger header.

Save only the existing native key encoding at a complete boundary:

```python
saved_motor_rng = save_motor_rng(motor.policy._rng)
# On restore, decode the original key and invalidate all ephemeral caches.
motor.restore_motor_rng(restore_motor_rng(saved_motor_rng))
```

Call `clear_cache()` at explicit reset/recovery/qualification boundaries.
KV is not serialized; pickling the adapter is refused. Preserve the existing
optimizer, sampler, task cursor, raw corpus, partial-batch and physical-journal
recoveries exactly. The adapter supplies no source reservation, simulator or
learner and performs no process/allocation control.

Current source SAC records frozen VLA features before decoding; its official
actor/replay path uses the separate `sac_observation` pixel adapter. The motor
cache does not change that SAC encoder or replay computation. PPO's current
`run_episode` and RL2 callback make repeated frozen-feature reads of the same
current observation before decoding. A separate one-entry host feature cache
preserves feature reuse after a successful decode without reusing consumed KV;
this is interface compatibility, not a measured source-SAC optimization.

## Required runtime gate

Use identical current observations, actual instructions and explicit noises
for the original and cached paths. Keep the original feature maximum2e-4 and
action maximum1e-5 gates. Verify every record, strict shapes/finite values,
once-per-decode identical native motor RNG progression and unchanged full model
parameter hashes. Make a cached feature call before each cached decode; the
initial native-fallback infer calls alone do not qualify the cached path.

The gate must include actual PPO and SAC call orders, bounded PPO coordinates and
unbounded SAC warmup noise, all available source suites, cache identity errors,
qualification/reset/complete-boundary restoration and serial replay. Report
model-path time separately from loading, compilation, input I/O and audit
overhead. A matched recorded-observation comparison establishes kernel/feature
compatibility; it does not establish live collection throughput, reset identity
on additional cases, PPO improvement or robot adaptation.
