# Source100 AD-first fit and robot readout

Proposal only; no admission, resource reservation or launch. This is the next
bounded development phase, not completion of the larger benchmark matrix.

- Existing implementation owner: native RoboTwin history exporter/evaluator.
- Reuse the qualified exporter, trainer, controller and real robot loop.
- AD/BC-only routing is committed at
  `7401cd50c038a4cb35149eceeb72128e9a368cc3`. Focused regressions are prepared
  but still require allocated Linux execution. The source100/500-update
  evaluation profile is being prepared by the same owner. Pin/review each
  component's resulting commit before its execution.
- No new controller, source collector, teacher panel or benchmark adapter.
- AD labels: actual executed steering noise. Validation: behavior cloning on
  actual held-out source episodes. No mean-teacher competence requirement.
- DPT remains a later conditional arm; preserve its teacher-label checks and
  the existing paired default. This phase does not silently redefine DPT.

## Fixed data and fits

- Source: all600 SAC K4 trials /2400 episodes /32425 decisions /1003790
  source actions, including warmup/failures, in actual chronological order.
- Source100 snapshot SHA:
  `59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90`.
- Closed validation ancestry: original root plus slices0001–0061; use the
  owner's bound ancestry list, not an inferred/unverified directory append.
- Final coordinator SHA:
  `f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8`.
- Validation: source100's6 K4 /24 episodes, actual behavioral targets. Enforce
  case/episode separation; no target trajectories enter the bank or fits.
- AD full and AD static, seed0,500 updates each;1000 total updates.
- Both fits reuse the same source bank; count its physical collection once.
- Existing recipe: batch8, max-window128, width128,2 layers,4 heads,
  context300, LR3e-4; preserve actual valid-token weighting.
- AD uses Gaussian sequence NLL with a learned1024D log_std vector. Primary
  source100 readout samples that learned Gaussian and applies the existing
  physical bound1.5, using the controller's private generator. This is fixed
  before any new robot outcomes. Legacy mean-mode results stay separate;
  this phase adds no mean-only condition.
- Save25/100/500 full optimizer/sampler/RNG recoveries and source curves.
- Primary weights: candidate-000500.pt, fixed before target readout.
- Static training and deployment both erase prior noise/reward/cross-step
  cache while retaining current features and real episode/primitive clocks.
  It is the same transformer family, not a memoryless MLP architecture.
- Qualify the new objective route and final100 bank on allocated CPU before
  GPU fitting. Failed qualification spends no GPU-fit allocation.

## Initial robot readout

The existing two development goals only: adjust_bottle and move_stapler_pad.
The full six held-out-goal and0/1/2/4/8 protocols remain outstanding.

- Eight fixed query cases/goal. Requested query seeds:
  22261030,22262027,22263024,22264021,22265018,22266015,22267012,22268009.
- Declare one separate support case/query: support seeds32161030+997*i,
  i=0..7. Record accepted/resolved seeds and actual initial observations;
  seed differences do not prove physical-state novelty.
- K0: native, frozen sourceSAC, ADfull, ADstatic;16 queries each =64 episodes.
- K1: ADfull and the same ADfull weights with cache+previous-noise/reward
  cleared at the query boundary;16 actual support/query pairs each =64 episodes.
- Total:128 episodes,32 supports and96 scored queries. Score queries separately.
- Fresh controller context per pair. Fixed student and motor parameters;
  no query SGD or optimizer updates. Preserve within-episode recurrence.
- Independent declared support/query sampling streams. Reinitialize query
  sampling randomness without clearing FULL's support history; pair query
  draws between same-weight conditions. Preserve the real episode clock.
- Compare FULL and erased histories at the same K1 query clock. K0's fresh
  first-episode clock need not be byte-equal to a K1 reset query's clock.
- Match actual query seeds, raw inputs/features and motor/interface contracts
  across conditions. Report mismatches, aborted pairs and all interaction costs.
- Use the existing qualified native episode/process lifecycle and action masks.
  No new GPU namespace or alteration of an existing run.
- Record actual support and query costs separately. Sequential K8 episode
  indices or a four-support/four-query label cannot stand in for this test.
- Retain partial outputs/recovery and charge unresolved execution intents.
  An incomplete support/query pair is not a scored one-shot result.

## Proposed ceilings

These are fresh-phase requests, not unused balances from closed campaigns.

| Phase | Proposed resources | Ceiling |
| --- | --- | ---: |
| Source100 projection, AD/BC export, focused checks |0 GPU,8 CPU,64GiB|7200 CPU-worker seconds|
| Two500-update fits, including startup/validation/cleanup |1 qualified GPU|2100 GPU-seconds|
| Initial robot readout, qualification and cleanup |1 qualified GPU/worker,16 CPU,128GiB,8GiB shared memory|7200 aggregate GPU-seconds|
| Shared setup/failure/contingency |within the same new grant|1500 GPU-seconds|
| Total new GPU envelope |count each allocation once|10800 GPU-seconds =3 GPU-hours|

- New target-action ceiling:60000, including qualification/retries and all
  counted native actions; planned128*400=51200 before overhead. No new source
  interactions. Keep pending/uncertain charges conservative.
- Proposed new-storage ceiling:128GiB, with existing raw banks immutable.
  R6 measured56.8GB of unique input bytes; full-source read estimate87.4GB.
  Read volume is not assumed to equal new storage; do not copy raw banks
  unnecessarily. R6 AD NPZ716MB scales to roughly1.1GB before validation and
  checkpoints. Check actual capacity/writeability in the allocation.
- CPU and GPU ceilings include startup/failure/cleanup. Nested stage clocks
  overlap their enclosing allocation and are not added twice.
- Software qualification is at most600s INSIDE the7200s aggregate CPU total,
  not an additional allowance. A separate software job is permitted: deduct
  its actual terminal scheduler charge, then reserve only the remaining CPU
  time for the final bank. Each parent starts before its own probe/tests/work;
  final-bank parent <=min(6900s, its scheduler reservation minus300s).
- The12 condition×task readout units execute serially as finite children
  inside ONE exclusive1GPU allocation. They are not12 scheduler reservations.
  The owner's current5400s scheduler/5300s parent proposal fits within the
  7200 aggregate readout ceiling. Review the exact resulting handoff/pin.
- For a serial single-allocation variant, the GPU parent is at most10500s
  inside10800s. For split allocations, use per-allocation margins and the
  aggregate ledger; do not substitute one wall clock for aggregate GPU cost.
  All margins stay inside the1500s shared setup/contingency allowance.
- Software components are now qualified, while both software wrappers remain
  Failed/exit1 (selector, then post-cleanup JSON publication). Actual139CPU-s
  is counted once; the old461s unused allowance is sealed. At most7061CPU-s
  remains only if the pending combined phase is approved. The new sampled
  readout's fixtures still need actual allocated qualification. See
  [SOFTWARE-OUTCOME.md](SOFTWARE-OUTCOME.md).
- Distinct sampled readout f7594a03cb8f0f683d366193ed278b7435a4378c now passes
  all48 allocated CPU fixtures in a Success/exit0 job,31 additional CPU-s.
  Aggregate software170 is counted once; remaining CPU capacity is7030 only
  after approval. Actual final100 bank/500 weights/native GPU readout remain
  separate runtime prerequisites. Earlier failed statuses stay unchanged.
- Do not borrow LIBERO's32h pool, its1180s remainder, sealed native-panel time,
  or any closed CPU qualification balance.
- No part of this phase is funded yet. The resource decision covers these
  fixed data, models, cases and ceilings. CPU checks and final-bank production
  come first. Funding alone does not waive code/profile qualification; each
  GPU component starts only after its exact handoff and prerequisites pass.

## Interpretation

The preceding pilot is not an ICRL success: matched historical native6/16,
FULL8/16, trainedRESET8/16, frozenSAC8/16, onlineSAC5/16. Source histories
show no strong sustained outer or within-K4 improvement. AD-first tests the
actual collected histories without pretending their mean policy is an expert.
A good source loss alone is insufficient; require the fixed robot readout.
No target-based checkpoint, teacher or data selection.
