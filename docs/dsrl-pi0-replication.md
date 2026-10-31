# Original Pi0 DSRL on LIBERO90 task57

`meta_rl.dsrl_pi0_replication` tests ordinary online SAC steering of the
original frozen `pi0_libero` motor. This is separate from the existing
[Pi0.5 adaptation](vla-task57-dsrl.md) and all source40/meta-learning work.
Task57 records have role `target-online-baseline`; source assembly is forbidden.

Keep the authors' DSRL `7f48937`, its OpenPI submodule `a6d2400` and LIBERO
submodule `87edbd1` clean. The original public S3 bucket now returns
`NoSuchBucket`. The canonical official GCS `pi0_libero` checkpoint is the
hosting fallback, with cloud MD5/size checks and per-file SHA256 receipts.
Byte equality to the unavailable S3 release is not independently established
and remains an explicit replication limit. Use that checkpoint's own
normalization assets and the
original Pi0 H50 prediction/execute20 interface,400-step episode horizon,
native `env.reset()` distribution, and the authors' shared training/evaluation
environment. Do not introduce fixed initial-state banks, extra settling,
scene selection, Pi0.5 substitution, or a source-trained SAC initializer.

The runner calls the unchanged upstream collection, replay relabeling, SAC
update and stochastic evaluation functions. Its loop retains replay>500
query-record warmup, batch256, UTD20,10 critics,32D noise repeated over50
slots,−1/query reward with terminal0, timeout bootstrap and .999^20 discounts.
Whole episodes use fixed behavior weights; full optimizer bursts follow.
Ten-query evaluations occur at exact10k optimizer-update counts inside each
burst. The initial native evaluation occurs just before the first update,
after collecting warmup, as upstream specifies. It samples independent
H50x32 Gaussian noise. Keep replay queries, primitive actions, evaluations,
optimizer updates, wall time and GPU time as separate units.

The copied control flow keeps the original Q-value visualization calls and
their RNG effects. Local logging omits remote W&B/media publication. Physical
capture observes existing resets, states, inferred noises/actions and executed
actions/rewards without additional physics or rendering calls. A pending reset
reserves400 actions durably; completed episodes settle exact cost. A process
failure retains its unresolved reservation. Same-input motor qualification
restores the motor RNG, executes no additional physical actions, and verifies
the actual renderer and model process on the allocated GPU. Hash all motor
parameters before and after execution.

Complete recovery includes serialized SAC TrainStates/Adam/target/temperature,
agent RNG, compact filled replay plus capacity/trajectory cursor, both pending
prefetch state and global sampler RNG, Python RNG, typed motor RNG and complete
loop counters. Snapshot the suspended-yield refill flag so resampling does not
silently change batches after recovery. Alternate checkpoint slots using the
published cursor, including repeated saves at the same episode number. The
recovery helper restores these fields; the current fresh-run CLI does not yet
launch resumed training. Do not start a replacement actor under an old identity.
Retain a separate policy/optimizer checkpoint and pre-panel RNG at every exact
evaluation boundary, including the initial native panel; these records accompany
the rolling full recovery slots and the physical episode traces.

Run the focused accounting/recovery qualification on allocated Linux CPU.
The initial suite includes physical reservation/failure and iterator chronology;
the populated-SAC check restores optimizer/replay/prefetch/RNG and compares the
next actual SAC update. When only checkpoint slot selection changes, rerun the
affected populated-SAC check and explicitly carry unchanged prior checks.

The launcher requires one actual exclusive Baige GPU, an allocation probe,
passed qualification with current source hashes, and a fresh output identity.
One seed is bounded at36,000 all-in GPU-seconds,1M simulator actions and64GiB
outputs. Reserve a full collection episode and a possible10-query panel before
beginning a new burst, and stop at a complete recovery boundary with time for
cleanup. All failed allocations and any continuation must share the same phase
ceilings. A new run identity does not replenish a scientific budget.

Runtime uses Python3.11.11, released dependency pins, CPU Torch2.6 and
MuJoCo3.3.1. Preserve DSRL's Pillow10.4 override against the submodule metadata's
>=11 request, and NumPy1.26 for Robosuite1.4.1; save the resolved package list.
These compatibility choices are explicit. A single-seed pilot cannot establish
the paper's three-seed aggregate or its sample-efficiency confidence interval.

Primary references: [released launcher](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_libero.sh),
[original control loop](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py),
[paper](https://arxiv.org/html/2506.15799v2#S5.SS5).
