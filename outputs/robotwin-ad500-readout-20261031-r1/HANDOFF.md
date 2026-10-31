# Source100 AD500 readout preparation

Prepared 2026-10-31 Asia/Shanghai. Static checks only: no model execution,
pytest, fitting, simulator actions, resource admission or launch occurred here.
The AD7401 export/fit candidate and teacher063d successor remain separate pins.
Readout preparation is not another gate on the two AD fits.

Explicit inference successor to frozen5f394066dc79ca61cb5cfd1a586b1c52783e7d07.
Before any target outcomes or admission, the method coordinators changed the
new AD500 primary to the learned Gaussian sample followed by the existing
physical clip. No extra mean arm, condition, physical case or resource is added.
The previous5f mean draft and earlierc744 bank draft remain historical/unadmitted.
Canonical support32161030+997*i and query22261030+997*i are unchanged. Different
seeds alone establish no physical novelty; accepted reset/capture hashes and
pairing remain required. Qualified legacy mean profiles and raw results keep
their original pins.

## Scope and topology

Use existing robotwin_history_live/evaluate/controller, native runtime and
per-chunk physical ledger loop. No new controller, wrapper, collector or data arm.
Two DEV tasks: adjust_bottle and move_stapler_pad; eight queries per task.
Order: native, sac100-frozen, ad-full-k0, ad-static-k0, ad-full-k1,
ad-full-k1-reset, with both tasks for each condition: 12 fresh motor workers.
K0 has eight independent queries; K1 has eight independent support/query pairs.
Support failures stay in the context and result. No query feedback crosses pairs.
The K1 reset uses the SAME full weights, clears cache/noise/reward/carry flag,
and keeps query clock1 in both arms. It is not declared equal to K0 clock0.

Query environment seeds:22261030 + i*997. Support seeds:32161030 + i*997.
Query action seeds:70261030 + task_index*100000 + i*997.
Support action seeds:80261030 + task_index*100000 + i*997, i=0..7.
Requested banks and accepted support/query cases must remain disjoint. Native
raw/feature/case references cover each query; K1full supplies the independently
rerun reset arm's support references. The reset arm checks complete support
proposal and per-decision support trace hashes before issuing its query. Each
support decision frames little-endian float32 input features and physical noise
before dispatch, then canonical actual duration/reward/termination/truncation
metadata and typed following features after the outcome. Equal episode totals
alone do not establish the same support history. A changed intermediate reward
or feature with equal proposals/totals still stops before the query.
Only action RNG is reseeded after support, preserving intended history.
Students sample mu(history)+exp(clamp(log_std,-5,2))*epsilon using the existing
private Torch generator (`deterministic=False`), then the existing controller
clips the physical proposal to[-1.5,1.5]. This deploys the learned variance from
Gaussian sequence NLL. The plan explicitly binds `student_inference` and
`student_action`; CPU admission, execution and resume verification require the
same rule. Native noise stays unclipped Gaussian; SAC100 samples its physical
tanh policy with independent per-query generators. Student support and query
streams retain their declared separate seeds. Full/reset support traces and
paired query innovations stay matched while the intended memory lesion remains.
No old sequential K8 or source-validation private RNG stream is relabeled K0.

## Artifacts and selection

plan.prepared.json is immutable topology, not an executable bound plan.
Its SHA256 ise98957126989a0cfa9acc3286879bbf90b6379a20617350eb1f33753f435571a.
After the source-only fits, make a new owner-hashed copy changing ONLY:
- models: actual absolute paths and SHA256 for ad-full/ad-static candidate-000500.pt;
- fit_commit: exact clean source-only fitting commit;
- train_index_sha256 and validation_index_sha256: actual qualified final100 AD/BC indexes.

No actual AD500 model or final100 export hashes exist in this preparation.
Never replace missing hashes with a fabricated checkpoint or target-selected
best-validation candidate. Existing load_policy checks physical file hash and
portable format; the new selector additionally checks update500, mode, AD/BC,
source59588, all600K4/2400E/1003790 source actions,6K4/24E/10180 validation
actions, disjoint source cases, fixed recipe and exact index/fit bindings.
Full/static source-only normalization must match. Source actor59588 is the
sampled SAC100 control; its mean competence is not an AD prerequisite.

## Resource proposal, not admission

Fresh K0:4conditions*2tasks*8queries=64episodes/25600 max actions.
Fresh K1:2conditions*2tasks*8pairs*2episodes=64episodes/25600 max actions.
Total128episodes/51200 max actions, zero SGD, no paid reuse. K0 unit reserve3200;
K1 unit reserve6400. One parity reset per worker adds12 zero-action resets;
max accepted resets per worker is17, plus bounded existing reset setup attempts.

Propose at most5400 scheduler GPU-seconds for the existing single-GPU readout,
parent wall5300, final reserve60, per-unit start reserve300. This planning cap
uses the existing same-sized readout envelope; it is not a measured throughput
promise, resource grant, retry allowance or access to closed/sealed budgets.
Execution owner must review it in the combined CPU/export + ADtwofit + readout
budget and use a fresh campaign/output identity, exact clean merged code pin,
actual host, probe, renderer/native qualification and original ownership guards.
Partial/unknown outcomes retain reservations and halt; no automatic retries.

## Prepared validation

48 CPU cases in tests/test_robotwin_ad500_readout.py are prepared, not run.
They use injected ledgers and environments, no production CampaignBudget import.
Coverage: topology binding, per-query isolation, actual failed support retention,
RNG-only reseed, equal K1 clocks/exact lesion, native/SAC sampling conventions,
case drift/overlap before acting,400step/final16 accounting, strict500/source100
selector,6400 pre-model allowance, child profile/reference/parent identity,
source normalization, unknown outcome reservation,12worker128episode topology,
complete-unit recovery and tamper rejection. New regressions exercise the real
small native policy at two learned standard deviations/full and static modes,
verify sample-then-clip against the declared private stream despite global RNG
draws, preserve context while pairing query innovations after unequal support
lengths, reject mean-plan/mean-agent/mean-receipt substitutions, and retain
same-weight stochastic support trace pairing. Two negative fixtures hold all
proposals and aggregate return/duration/chunks/outcome fixed while changing
intermediate reward or features; the per-decision digest must reject the lesion
before any query. A resumed support row also requires its trace digest.

For an allocated CPU gate under the existing no-native/no-ledger firewall,
select these48 plus existing evaluator/live/parent/source regression cases as
reviewed by the execution owner. Three legacy tests intentionally instantiate
real CampaignBudget and require their already qualified integration context:
test_child_closes_at_real_ledger_k8_boundary_and_no_step_occurs_in_parity,
test_child_exception_preserves_pending_reservation_and_stops_recovery,
test_insufficient_k8_allowance_never_loads_agent_or_prepares_runtime.
Do not weaken the import firewall or claim these integration tests ran there.
All passing counts must come from real allocated JUnit receipts. This readout
is not folded into the closed AD108/teacher57 component qualification:139 actual
worker CPU-seconds and461 sealed unused seconds close its600-second envelope.
No future readout test or runtime may borrow that sealed balance. AD7401 and
teacher063d component results remain immutable; the wrapper's publication failure
and missing end-to-end qualification remain recorded in the execution receipts.

The larger0/2/4/8/all-six-target protocol remains open. Target success is not
measured, target data select no endpoint, and no ICRL benefit is claimed here.
