# DPT-inspired CTM warm-start pilot

Approved scope: add supervised warm starts alongside direct full-trial PPO, for
compact benchmarks and the native visual pi0.5 bridge, with separate checkpoint
families. This is a bounded first-candidate pilot, not a claim of successful ICRL.

The durable method contract is `docs/ctm-warmstart.md`. Training histories keep
the actual executed actions and resulting feedback separate from the privileged
expert action labels. Both training routes replay complete chronological trials
with gradients across attempt boundaries. No trained recurrent teacher is needed.

## Allocation and stages

One Baige GPU, 16 CPU cores, 64 GiB RAM, at most 60 minutes. The native learner
uses the GPU; three two-thread workers train compact benchmarks on allocated
CPUs. The existing direct-PPO run is independently owned and remains immutable.

1. Run unit/interface tests and the actual compact CLI pretrain/resume/PPO test
   inside the scheduler allocation. Stop on a failure.
2. Save separate CTM candidates for bandit, Bernoulli, nonstationary bandit,
   hidden-gain reach, Dark Room, Key-to-Door and cached rotation. Each uses width
   64, four internal ticks, eight attempts, 256 training trials, 16 validation
   trials, batches of eight, and at most 200 updates/300 seconds. The grids use
   eight behavioral evaluation trials; other tasks use 32. Default horizons are
   unchanged. Record completed updates, rather than equating time caps to equal
   training budgets. Cache labels use only training-scene aggregate outcomes.
3. Collect two native batches of two complete four-attempt trials from the frozen
   noisy policy. At the end compare the inverse-rotation label policy with the
   frozen policy on paired training-distribution plans: four trials / sixteen
   episodes per policy. Require oracle success at least 0.25 and no lower than
   baseline before using the labels. This small gate is not a significance test.
4. If the label gate passes, fit the native CTM with conditional flow matching for
   at most 100 updates/300 seconds, reserving one complete history batch for
   validation. Save a distinct native warm start. Evaluate the frozen candidate
   under normal, memory-wiped and reward-blind conditions on held-out rotations
   and overlays, then perform one PPO update to qualify the initialization path.
   Report the evaluation at iteration zero separately from that PPO update.

Every output directory is new and pinned to a clean reviewed main commit. Every
candidate includes data checksums, benchmark/interface metadata, full cost
counters and evaluation; `latest.pt` additionally retains optimizer, sampler and
RNG for resume. The exported `warmstart.pt` transfers learned recurrent/actor
weights to fresh PPO with a zero critic and fresh optimizer; trial memory is reset.

## Interpretation

Tiny data and time limits are deliberate plumbing/learning screens. Falling
supervised loss alone does not demonstrate adaptation. Compare normal against
memory-wiped and reward-blind curves, report teacher data cost, and retain direct
PPO as the baseline. Fresh grid seeds do not guarantee disjoint latent grid goals.
Cached rotation is a scalar control, not a robot result. The native inverse-action
labeler is an engineering extension of the DPT/Vintix II recipe, not a reproduction
of either paper. A failed native label gate blocks native pretraining but leaves
compact candidates and all diagnostic evidence available.

## Qualification history

- `9945ddb`: first command was malformed by the web editor and stopped; no model
  ran. The next submission failed its GPU visibility precondition before tests.
- Baige `job-fg109ot4yftb`, output `ctm-dpt-qualification-20261028-r2`:
  GPU probe passed; 63 tests passed, 9 Slurm-only tests skipped, 2 new tests failed.
  Both failures were test defects: float32 continuous-action round-trip tolerance,
  and a tiny CTM fixture requesting unsupported synapse depth 1.
- `151b366` fixes those fixtures and adds a real allocated-Linux warm-start CLI
  resume/initialization test. `349660f` incorporates the latest main changes
  (the independent CMM decay-initialization option) before requalification.
- Baige `job-5rd5ul04uwto`, output `ctm-dpt-qualification-20261028-r3`, source
  `349660fe4b4a75299d42036af9ad4e1cfa83a304`: **66 passed, 9 skipped in 15.97s**.
  The nine skipped tests require a Slurm step; the new CLI qualification ran on
  Baige and passed, including bitwise equality of resumed and uninterrupted
  warm-start weights/optimizer, then successful initialization of one PPO update.
  The real allocation's CUDA probe passed. Evidence is retained in `evidence/`.
