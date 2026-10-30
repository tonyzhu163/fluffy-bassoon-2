# Six-case ordinary DSRL learning-onset baseline

This bounded target-online baseline reuses the authors' pinned DSRL
`PixelSACLearner` through `OfficialSAC`. Each of LIBERO90 task IDs0,10,20,30,40,50
gets a **fresh** actor/critic/temperature, replay, optimizer and sampler.
Every case uses state0 and scene seed123, declared before outcomes. No target
interaction enters a source assembly or meta-controller pretraining.

The motor is the same official frozen OpenPI40 π₀.₅, native H10/execute10,
with32D noise repeated over its horizon. Warmup is unbounded tied Gaussian;
learned noise is clipped to[-1,1]. Weights and behavior stage stay fixed
within each episode. After a complete episode, replay>500 queried chunks
enables batch256/UTD20 updates; timeout transitions bootstrap and genuine
successes terminate. Actual primitive-duration discounts are retained.

**Representation differs.** Canonical authors' SAC consumes64×64 single RGB
plus8 proprioceptive values through its CNN. The meta controller consumes
4104D frozen native prefix features. This driver records those exact4104D
features in raw training/query traces but does not feed them to the SAC actor.
It is an ordinary released-code baseline with an explicit representation
confound, not a feature-matched SAC variant or a claim of algorithm parity.

One separate frozen query follows0/1/4/8/13/16 training episodes. Queries use
the same case and fixed query random stream, never add replay rows or perform
updates, and restore actor/motor/global RNG afterward. Before any update,
published evaluation uses native independent H10 Gaussian; afterward it uses
stochastic SAC actions. These are held-out **episodes**, not unseen physical
placements. The13-episode milestone captures earliest possible warmup
clearance when every training episode lasts400 actions/40 chunks. Easy cases
can remain below500 chunks after16 episodes; report zero updates honestly.

| Ceiling / count | Value |
| --- | ---: |
|Training episodes across six cases |96 |
|Separate frozen query episodes |36 |
|Per-case action cap, including resets/queries/retries |9,500 |
|Worst-case16 training+6 query episodes+initial qualification |9,030/case |
|Nominal complete six-case cost |54,180 actions |
|Sum of six case caps |57,000 actions |
|Joint target action ceiling |64,000 actions |
|Requested maximum additional allocation |One GPU, at most2h |

The7,000-action gap above summed case caps is unassigned; no case may silently
spend it. Extra qualification resets and unresolved interrupted RPC intents
count inside the same case cap. The outer authorized allocation, root's
32-GPU-hour ledger and this run's cumulative `worker-time.json` govern time;
`--resume` does not grant another2h budget.

Deploy the reviewed integrated commit into a new immutable run worktree, then
inside an already approved one-GPU allocation:

```sh
bash scripts/diagnostics/vla_target_online_dsrl_six.sh \
  <exact-SHA> <fresh-run-id> 7200
```

The first invocation reserves7,140seconds plus60seconds scheduler allowance.
It starts the cumulative clock before GPU setup and tests; finished failures
charge measured runtime plus the same allowance. An unclosed attempt remains
charged at its full reservation. A resumed invocation receives only the
remaining slice less scheduler allowance; fewer than600seconds is rejected.
The fixed7200-second scientific ceiling and action/recipe/case contracts never
change with invocation length. The driver's explicit `--stop-unix` is120seconds
before the reserved slice end; its outer timeout is60seconds before that end,
leaving another60seconds after the soft driver stop for orderly shutdown.
An overrun is reported at its actual duration rather than truncated.

The launcher probes the actual GPU and runs only relevant baseline, physical
journal/warmup and cumulative-clock Linux tests before model interaction.
Typed and legacy motor RNG round-trips run in the actual model environment.
It preserves per-attempt logs/probes/tests/clock receipts. Model computation
remains on allocated Linux. For an interrupted run, reuse the exact source/run,
keep the fixed7200 argument and append `--resume`; the clock automatically
chooses a shorter remaining slice. A terminal scheduler receipt may justify
explicit reconciliation of an unclosed attempt by its owner; never delete or
reset the time ledger to obtain another allocation.

Per-case recovery includes actor/critic/target/temperature and all optimizers,
full replay, sampler, Python/NumPy/agent RNG, typed motor RNG, target training
cursor and milestone cursor at a complete episode/update boundary. Raw RPC
inputs/outcomes are journaled before learning; unresolved actions are charged
conservatively and never fabricated as completed replay. Milestone snapshots
are immutable. Ordinary DSRL convergence, full-suite scores and relative
sample-efficiency claims remain unestablished by this learning-onset slice.
