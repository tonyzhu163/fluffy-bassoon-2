# AD source100 fit-to-robot readout preparation

Static review,2026-10-31Asia/Shanghai. No allocation, physical actions, model
execution, tests, fit, new monitor or resource admission occurred here.
Sole native/export/trainer ownership remains with the existing exporter chat;
root owns combined execution profiles, admission, scheduling and accounting.

Existing `robotwin_history_evaluate.run_trial`, `StudentAgent`,
`FrozenNativeHistoryController` and native live entrypoint are the implementation
path. At frozen26dd63d, `load_policy` accepts native-format500-update weights;
controller lifecycle/observations/outcomes and per-chunk physical accounting
already exist. Extend the strict plan/profile selector rather than introduce
another controller, environment bridge, collector or wrapper.

Prepare the AD-only export/BC-validation and full/static500-update fits first.
Mean-teacher competence and a source-teacher robot panel are not prerequisites
for executed-noise AD/BC labels. DPT-specific competence/inference guards remain
conditional on DPT; old paired/four-fit defaults remain available and unchanged.
Readout preparation proceeds in parallel and does not become another fit gate.

The first bounded readout uses the existing two development goals and sixteen
fixed query cases: native, frozen source100SAC, ADfull and ADstatic atK0; ADfull
and the same ADfull weights with a support-history reset atK1. Predeclare a
separate disjoint support bank and independent, paired query action RNG. Keep
failed support histories and actual native costs. Checkpoint500 remains primary,
selected by the source-only recipe; targets select no checkpoint, teacher or bank.
This limited readout does not fulfill the larger0/2/4/8/all-six-target protocol.

K0 must start a fresh controller trial for each query. K1 must start a fresh
support-to-query pair for each query unless a separately reviewed exact support
state fork/replay is used. Query outcomes may never become later support.
Reseeding query action RNG must preserve the intended support cache/feedback;
calling reset_trial merely to reseed would erase the treatment.

The existing controller's reset lesion clears recurrent state, previous noise
and reward, and changes the episode-boundary carry flag. It retains sequential
episode clocks: a K1query is atindex1, while a K0query is atindex0. Keep equal
query clocks across K1full/reset and document the exact lesion. Do not require
false byte-equality between K1reset and K0 or infer memory benefit from clocks.

A fresh, unreused upper bound for the stated topology is64 K0query episodes
plus2conditions x16queries x(support+query)=64 episodes:128episodes/51,200actions
at400-step limits. This is a planning bound, not an admitted resource budget.
Existing native16/SAC16 or support-state reuse may reduce it only after exact
actor/motor/capture/case/private-query-RNG and once-counted-cost qualification.
A prior sequential K8 student history is not automatically a bank of K0queries.

The owner supplies exact code pins, topology, meaningful CPU regressions,
source/fit/model bindings, commands and costs. Root then assembles one concrete
CPU-export/software-qualification + two ADfits + robot-readout budget proposal,
with real allocation probes and fresh separate identities/ledgers. Existing
closed/sealed reservations and the stopped original campaign monitor supply no
new capacity. No resource request is ready for execution from this note alone.
