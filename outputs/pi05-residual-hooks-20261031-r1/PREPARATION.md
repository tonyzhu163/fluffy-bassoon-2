Update 2026-10-31: 41 synthetic CPU tests passed at source4cbc96a; actual VLA/cache/simulator qualification remains pending. See [CPU result](../pi05-residual-hooks-cpu-20261031-r1/RESULT.md). The original preparation record follows.

# Reactive residual implementation preparation — 2026-10-31 Shanghai

Status: **UNQUALIFIED implementation; no allocated tests or live screen run**.

Reviewed remote main through the GitHub connector at
`fd4a7c91207210951e5bbfc3ecb80c61fedd9052`. Created a new managed detached
worktree from that exact revision:
`/Users/mowenjie/.codex/worktrees/pi05-residual-hooks/fluffy-bassoon-2`.
The original owner checkout and all active/ended runtime worktrees remain
unchanged. Draft contract read from the redesign owner's
`outputs/frozen-pi05-steering-screen-20261031-r1/PLAN.md`.
Refreshed main before commit to `c5db2a7e198437e56bb5618cf0b4515c47c60d50`.
The connector comparison shows only the redesign plan, saved headroom audit
and coordination metadata changed in its three new commits; none of this
patch's source/test paths changed. The implementation parent remains the
reviewed `fd4a7c9` code, compatible with those metadata-only successors.

Prepared a reactive low-rank native KV editor, a bounded decoded 14D ALOHA
head, a differentiable frozen-suffix primitive, optional scalar-backend
integration, typed intervention metadata and focused synthetic tests.
The older feature/velocity bridge is unchanged and is not called KV editing.
The native backend path remains the default. No learner, launcher, wrapper,
monitor, source/target selection, corpus rewrite or resource reservation was
added. New modules load no real model or simulator.

Review feedback incorporated: the training helper rejects no-grad and
inference-mode callers; action-head configuration includes its width and
separate decoded arm/gripper bounds. A detached suffix result also rejects a
runtime that cuts the adapter's gradient path. Public-feature/prepared-input
bindings fail closed if a caller restores stale lane caches. Parallel lane
support is not implemented or qualified.

Validation performed on macOS: Python AST parsing of all changed Python files
and Git whitespace/diff inspection only. No module import, Torch computation,
pytest collection/execution, model, simulator or GPU probe was run on macOS.
The new file contains eleven test functions (nineteen parametrized cases,
statically counted; none collected or executed). They cover zero parity,
masks/layout/padding, input nonmutation,
low-rank edits, motor/input gradient isolation, nonzero adapter-output gradient
through a frozen suffix, disabled/detached-gradient rejection, ALOHA bounds,
command clipping, native backend placement, metadata and executed durations.

Concrete testing limit: the current host is Darwin, no actual allocated Linux
resource or fresh CPU-only qualification identity/ledger is assigned to this
task, and the delegating owner reports no configured school-cluster aliases.
Both local `proxy_on`/`proxy_on_clash` shell functions are also unavailable.
Remote main review used the connector; local-only Git inspection/commit needs
no network routing. No scheduler query, job dispatch or closed-grant reuse was
attempted. CPU qualification remains pending a genuinely available allocated
Linux route with its own bounded admission; static checks are not test passes.
The delegating redesign owner is coordinating any separate CPU qualification;
this chat will neither dispatch nor search for an allocation. The affected
existing backend regression file is `tests/test_robotwin_noise.py`.

Future focused command inside that allocation:

```sh
uv run pytest tests/test_pi05_residual.py tests/test_robotwin_noise.py -q
```

Even successful synthetic CPU tests would not establish actual checkpoint/
normalization/cache compatibility, decoded native parity, a real episode,
teacher competence or a steering benefit. Those remain separately admitted
future work. DSRL-first/native-PPO-later priority and all existing budgets are
unchanged. No new GPU, simulator, robot action or SGD cost was incurred here.
