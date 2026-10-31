# A1 executable implementation handoff — 2026-10-31 Shanghai

Status: **new driver/mappings awaiting allocated CPU qualification and actual
GPU qualification**. The existing hook-only code4cbc96a had independent
41-test Linux CPU PASS; that result is not transferred to this new executable.
No GPU/simulator/robot/SGD/CPU test allocation or scheduler/CUA operation was
performed by this implementation chat. Resource submission belongs solely to
the existing Plan VLA in-context learning owner.

Started from reviewed main `fab7d03566b72a7eda27bc2eb2cf0b229d7da822`, after
rebasing the existing detached worktree and preserving main's independently
updated CPU evidence. Read the coordinator's SCOPE.json and INTERFACES.md in
`outputs/frozen-pi05-endtoend-20261031-r1/`. They authorize ONE targeted pilot,
not another per-child8h or an old campaign/closed CPU allowance. Driver uses
the exact supplied six-lineage canonical header and shared caps; creates no
resource grant or scheduler job. Source role/method IDs and hidden/public
command boundaries were agreed with the condition/collector and student owners.

Files: `rl/robotwin_steering.py` (four-coefficient mappings/native wrapper),
`meta_rl/robotwin_steering_screen.py` (qualification, source A1 search,
complete-episode API, recovery and admission), existing `rl/robotwin_noise.py`
(optional post-command and requested-command fields only), diagnostic CLI,
focused tests and `docs/robotwin-four-coefficient-screen.md` (exact API/input
schemas/command/boundaries). No second learner, corpus logger, simulator or
budget ledger was introduced. NoiseSAC and all old defaults remain unchanged.

Source candidate horizons are read from the verified RoboTwin YAML. For actual
400/400, worst case is `8*(400+400)+3*[3*(3+2)+2]*400=26800` actions across
67 complete episodes, including qualification, three arms/candidates, two
confirmations per winner and two matching native controls per case. A different
native horizon produces a recorded smaller case count before outcomes or
rejects the grid; episodes are never cut. All32 locked query cases stay closed.
Unused3200 is not more search. A1 outcomes are descriptive restricted-class
reachability, not universal feasibility, held-out gain, teacher expertise or ICRL.

Physical-map support issue resolved before any new test/live use: final mapping
is `arm_bound*tanh((temporal*c)@B)`, no divisor, constant basis row only on
arm joints0/1/7/8; grippers6/13 exactly unchanged. KV/noise keep their linear
bounded-delta `/4` mapping. Exact bases/hashes are in runtime config/recovery;
aligned calibration and candidate odd inverse remain controlled A2 diagnostics
requiring actual decoded/actuator/limit proof. They cannot rank natural A1 arms.

The driver verifies SHA-bound motor/norm/source conventions and source URDF
joint limits rather than treating a bare receipt digest as range proof. It
requires a current actual1GPU probe/ticket, source-bound zero reset-policy
actions, fresh output under shared64GiB, original absolute clock and one
shared cumulative action/GPU envelope. Actual terminal scheduler cost is
settled by the resource owner, separate from parent/episode wall times.
Closed operational boundaries alone resume; pending/uncertain/failure states
are preserved and block automatic retries. Full fixed bases, cursor, global
and private RNG, raw roots and closed episode SHA refs survive recovery.

Static checks on macOS: AST parse of changed Python sources/tests and Git
whitespace checks. No module/Torch import, pytest collection, functional test,
model or simulator execution occurred on macOS. Fresh allocated Linux CPU
qualification command (inside the coordinator's separate<=1800worker-s grant):

```sh
python -m pytest -q tests/test_pi05_residual.py tests/test_robotwin_noise.py \
  tests/test_robotwin_steering.py tests/test_robotwin_campaign_budget.py
```

Retain raw logs/JUnit/allocation and code hashes. Then the sole resource owner
may deploy the pinned executable into its separately admitted first stage;
actual zero/native decoded parity, true frozen-suffix gradients and a complete
real episode must pass before any qualification claim. No extra approval loop
or allocation from this chat is needed.
