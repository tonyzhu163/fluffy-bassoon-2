# RoboTwin trained reset controller: iteration-100 endpoint

Independent read-only verification, 2026-10-31. This receipt covers only
`ppo_reset`. The full-memory and SAC final endpoints are outside its scope.

The saved controller reached 100 complete source iterations: 600 complete K4
trials, 2,400 episodes, 963,759 source actions and 500 optimizer steps. Its
declared source cap is 1,100,000 actions. The terminal canonical view charges
1,021,785 total actions: 963,759 source, 39,434 source validation and 18,592
target. Pending and uncertain charges are both zero. Evaluation actions remain
separate from the source-learning exposure. These totals are bound to the
saved owner audit and independently rehashed canonical snapshot; the optional
full 66,181-event replay was stopped before completion and is not presented as
an independent event-by-event reconstruction. The 2,400 source episodes follow
the saved 600 complete-trial count and the four-episode K4 contract, not a new
raw-corpus episode scan.

| Fixed source iteration | Source actions | Bottle adjustment | Stapler movement | Source validation |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 94,876 | 6/8 | 1/8 | 4/24 |
| 25 | 239,436 | 7/8 | 0/8 | 4/24 |
| 50 | 479,895 | 7/8 | 0/8 | 2/24 |
| 100 | 963,759 | 8/8 | 0/8 | 3/24 |

At iteration 100, bottle queries executed 1,429 primitives and stapler queries
executed 3,200. Source validation was 3/24 across six source tasks. Full
per-stage source-validation counts, per-episode outcomes, primitives and
bounded/truncation fields are retained in `receipt.json`. The fixed stages
were reported without selecting a checkpoint from target outcomes.

The target seed bank is `22261030, 22262027, 22263024, 22264021, 22265018,
22266015, 22267012, 22268009`. Actual initial-observation hashes are retained
for each case in the receipt. Matching reset seeds and initial-observation
hashes across stages establish the saved physical observation match; they do
not establish identical controller RNG states or a shared process across
source stages.

Five of six source-validation goal banks also match reset seeds and actual
initial-observation hashes across stages. `handover_mic` uses the same four
seeds but different initial-observation hashes across stages. That source task
is not asserted to be an identical-observation paired comparison. The receipt
retains every differing hash without inferring its physical or rendering cause.

The referenced prior native bank has 12 individual episode rows: before/after
probes at stages 10/25/50 for the two target tasks. Its seed is `21161030`, and
its initial-observation hashes differ from the reset bank. Those native probes
are separate context, not a paired native control. The earlier SAC5 native
panel is not pooled here. The reset endpoint itself contains no native rows.

All reset evaluations explicitly reset memory and episode feedback, with
reward blindness disabled. Thus the bottle result does not establish a memory
benefit. The stapler failure and low source validation also preclude a broad
robotic-performance claim from this endpoint.

The frozen motor record reports unchanged fingerprint
`721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41`.
The final coordinator recovery file is bound to
`48d910980052ce2cec730669fbb6432b86d956aadbff22799b7ecefdeaadcb49`.
The final summary is bound to
`8f5d5e2e9377df90fa955049e4464abf7ba58daa2977046902917b0132d1ff60`.
Stage checkpoint references and their independent byte hashes are in the
receipt. All nine supplied reference hashes matched across 26 rehashed saved
files; the canonical header content hash also matched. Every stage evaluation
file matches the terminal summary, all 32 requested evaluation groups have
their full episode count, and none is marked budget-truncated. Failed target
episodes reached the declared 400-action horizon; source validation uses its
declared per-task 400/500/600 limits. This verifier reads JSON and hashes file bytes; it does not deserialize
Torch recoveries, construct models, rerun inference/tests/simulators, or control
jobs/resources. Checkpoint existence and hash integrity are not a new restore
test.
