# Native pi0.5 headroom screen followed by ordinary DSRL

Pre-outcome executable amendment, 2026-11-01, reviewed main
`4e422587484312bfe34d623c5b498ab303a96be0`. This explicitly replaces the prior
3b71 draft's native+tied384-episode screen, broad confirmation and separate
outcome-blind10-query language. No fresh outcome was available when this
amendment was fixed. Follow-on interface ideas remain separate in NEXT-INTERFACES.md.

The human request is to find a task with weak starting native pi0.5 success,
then test whether ordinary online DSRL improves it. Entire original official
OpenPI LIBERO40 pi0.5 remains frozen (ec064 receipt, c565 motor parameter hash),
H10/execute10. No motor change, SFT, pi0 download or RLinf130 substitution.

Closed candidates are **0,29,33,38,44,53,59,64,70,77,79,82**. They were already
listed in the pre-existing3b71 plan, using saved official40-motor fixed100 K0
rows and older native cross-run variation. Source files and SHA256 values are
bound in executable plan.json. Fixed100 native was28/90 on one physical case
per task. Earlier native waves remain separate. Controller outcomes only
motivated candidacy; they are not pooled as native episodes or success rates.
There is no new semantic filtering or assertion of novel task goals.

Screen native ONLY:20 episodes per candidate, states0..19, fixed physical
reset seed123;240 episodes total. Private motor/noise seed derives from the
case ID and draw, never changes the physical reset seed. Native uses independent
H10×32 Gaussian noise through the qualified explicit-noise path. The screen
constructs no SAC actor, critic, replay records or optimizer.

Eligibility is2..8 successes/20 (10–40%). Rank by absolute distance from4,
then lower task ID. Confirm up to the first3 ranked candidates on a SEPARATE
fixed bank: states40..49×2 private noise draws =20 episodes,10 state IDs.
**Confirmation must also score2..8/20**. Choose the first passing candidate;
otherwise abort before SGD. Publish every candidate, denominator, failed
confirmation and physical/noise exposure. Record actual initial-state hashes
and their distinct counts rather than assuming seeds create new placements.

Freeze chosen task, full query bank, case seeds and plan SHA in
frozen-selection.json before ANY SGD. This is target-online development exposure;
no screen/query/learning trajectory enters source40 meta pretraining. No task
switch or favorable checkpoint choice is permitted after RL starts. Query
states are exposed for headroom selection and are not described as outcome-blind.

Learning uses the qualified authors' PixelSAC: fresh seed0 actor/replay/Adam,
10 critics, batch256, UTD20, replay>500 query-record warmup, complete episode
collection at fixed weights then updates, −1/query and0 on success, timeout
bootstrap, .999^actual executed duration. Native motor stays frozen. Train
states0..39; fixed query bank40..49×2 stays physically disjoint, checked against
actual nominal state hashes. Initial matched controls are native full Gaussian,
unbounded tiled32D Gaussian, and initialized bounded SAC with0 updates. All
are frozen, with shared physical reset identity and separate saved private seeds.

Endpoint500000 optimizer updates, after a complete burst (at most780 overshoot).
Twenty-query learned panels run after the first complete burst crossing each
20000-update threshold,25 panels total. Report actual update counts. Initial
and final native panels use the same20-case bank. This reduced panel frequency
is predeclared to fit the all-in action ceiling. The fixed endpoint is primary;
all milestones are reported, never selected for a favorable outcome.

All-in ceiling1M simulator actions includes screen, up to3 confirmations,
qualification, settling, controls, training, evaluation, failures and retries.
Loose worst-case plan:98400screen +24600confirm +24600initial controls +510780
training/settling +205000learned panels +8200final native =871580, with128420
reserved inside1M. Full chronological raw observations/noises/actions/rewards,
physical intents, two-slot full optimizer/replay/RNG recovery and immutable
milestone references are retained. Output ceiling64GiB; stop honestly before
whole-episode time/action/storage reservations fail, without deleting evidence.

Each actual job reserves at most7h, strictly below the human-authorized8h per
job; this is NOT an8h campaign cap. A finite16h cumulative phase accounting
ceiling includes all execution slices/setup/retries; native screen has its own
3h cumulative ceiling inside it. Subsequent jobs resume, rather than cold-start.
Screen can launch as soon as its focused allocated-Linux qualification passes;
trainer code follows through explicit source migration preserving the closed
screen plan/selection hashes and accumulated costs.

```
bash scripts/diagnostics/vla_target_online_dsrl_headroom.sh SOURCE RUN screen 7140
# If the same screen needs another slice:
bash scripts/diagnostics/vla_target_online_dsrl_headroom.sh SOURCE RUN screen 7140 --resume
```

Only finish_assembly owns Baige deployment, CPU tests, probes, simulator, model
execution and scheduler. The implementation author performs static checks only
on macOS. Focused tests: tests/test_vla_target_online_dsrl_headroom.py; existing
qualified task57 helpers remain unchanged. A first real native episode and whole
motor before/after parameter hashes remain allocation evidence.
