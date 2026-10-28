# Optional Meta-World environment workers

The default collector steps native environments serially. The optional
`metaworld_workers` backend groups lanes into persistent CPU subprocesses and
dispatches all groups before receiving results. Policy inference and optimization
remain in the parent process. This is an execution optimization, with admission
conditional on exact trajectory parity with the serial simulator.

## Interface

`make_worker_lanes(pool, count, benchmark=..., attempts=..., horizon=...,
workers=N, allocated_cpus=..., timeout_seconds=120.)` returns environment-shaped
lane proxies. Use `batch_reset(envs, seeds)` and `batch_step(envs, actions)` in
the collector. Both helpers return ordinary lists in the caller's lane order
and fall back to the native serial methods for ordinary environments. Calling
each proxy's `step` immediately waits for its result, so a per-proxy loop does
not provide parallelism.

The collector integration is limited to replacing its reset comprehension with
the first element of each `batch_reset` result and its step comprehension with
`batch_step(envs, action.cpu().numpy())`. Construct worker lanes only when the
optional backend is selected. The training and evaluation paths must both use
the selected backend, and their worker configuration belongs in the run manifest.
The backend does not alter models, losses, batches, trial lengths or task pools.

Each subprocess owns one stable strided group of original lane indices. It
constructs the same `MetaWorldTrial` wrapper, factory and ordered family task
pool as the serial path. The multiprocessing start method is always `spawn`,
so a CUDA-initialized parent is never forked. The worker target constructs only
CPU simulators and never receives a policy, optimizer, CUDA tensor or recurrent
state. Spawn may re-import the caller's main module; its module guard must
prevent training from running in children.

Worker count must be positive and cannot exceed the number of lanes, explicitly
allocated CPUs, `SLURM_CPUS_PER_TASK`, or process CPU affinity when available.
An explicit allocated CPU count or Slurm CPU declaration is required; host CPU
count alone is insufficient. Reserve CPU capacity for parent policy work when
choosing the worker count. The number of simulators remains the declared lane
count; adding workers does not add data or trials.

Proxies expose the common observation/action spaces, family, trial settings,
success outcomes and native returns. Their selected task and boundary state
are synchronized after each reset or physical episode boundary. Task identifiers
and process metadata never enter observations. Large task bytes are sent only
at boundaries to keep per-decision IPC small. `envs[0]._worker_batch.manifest`
contains backend, start method, worker count, CPU limit, lane grouping, process
IDs and timeout for raw run evidence.

## Failure and cleanup

Worker exceptions include their original type, message and traceback in a
`MetaWorldWorkerError`. A worker failure, exit or request timeout closes the
entire batch: some lanes may have advanced before another failed, so its state
must not be reused. The common trainer should preserve its preceding complete
learner checkpoint and record a failure receipt. Resume through a new output
identity and reseed fresh lanes from that checkpoint's sampler state.

Use `close_lanes(envs)` for explicit cleanup. Existing per-environment cleanup
loops are also safe: closing any proxy closes its owning batch and subsequent
closes are no-ops. Only child processes created by that coordinator are stopped.
Graceful close is bounded, with termination used for those owned children if
they do not exit. A process-exit cleanup handler is a fallback, not a replacement
for the trainer's `finally` block.

## Qualification and timing

Run tests only inside an allocated Linux step:

```bash
uv run pytest -q tests/test_metaworld_workers.py
FLUFFY_METAWORLD_REAL_QUALIFICATION=1 \
  uv run pytest -q tests/test_metaworld_workers.py -k real_ml10
```

The real test compares all ten ML10 training families, K2 H500 trajectories,
observations, scaled rewards, termination/truncation flags, success information,
executed action feedback, ordered task-pool digests, selected task digests,
native returns and success outcomes. It covers physical episode resets and a
subsequent independently seeded reset and transition. Fake simulator tests
also exercise persistent process identity, reordered/subset lanes, CPU bounds,
error propagation, invalid mixed batches and clean shutdown.

After parity passes, time serial and worker collection on the same pinned
source, hardware, policy weights, lane seeds and actions. Include subprocess
startup, IPC and cleanup in total allocated GPU hours, while recording startup
and steady collection separately. Compare two or more worker counts under the
same CPU allocation; increasing workers can lose time to IPC for cheap steps.
Keep serial as default until a real measurement supports enabling the backend.
This document reports no speedup or learning result.
