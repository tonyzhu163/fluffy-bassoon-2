# Independent frozen RoboTwin panel audit

**PASS: all 18 units completed; 22,135 independent artifact checks found zero errors.** The wrapper exit code and every child exit code are 0. The scheduler state was not independently queried; wrapper completion is verified separately from scheduler accounting.

This is one paired development K8 trial per goal and condition, at the predetermined first-fit endpoint100. It does not establish ICRL or statistical generalization.

## Success counts

Each goal has eight episodes. The checkpoint bits below are the individual episode outcomes at E1/E2/E4/E8, not cumulative success rates.

| Condition | adjust_bottle | move_stapler_pad | Bottle E1/E2/E4/E8 | Stapler E1/E2/E4/E8 |
| --- | --- | --- | --- | --- |
| native | 6/8 | 0/8 | 1/1/1/1 | 0/0/0/0 |
| native-zero | 6/8 | 0/8 | 0/1/1/1 | 0/0/0/0 |
| sac5-mean | 5/8 | 0/8 | 1/1/1/1 | 0/0/0/0 |
| ad-full | 5/8 | 0/8 | 1/1/1/1 | 0/0/0/0 |
| ad-static | 5/8 | 0/8 | 0/1/1/1 | 0/0/0/0 |
| dpt-full | 5/8 | 0/8 | 0/1/1/1 | 0/0/0/0 |
| dpt-static | 6/8 | 0/8 | 1/1/1/1 | 0/0/0/0 |
| ad-full-reset | 6/8 | 0/8 | 1/1/1/1 | 0/0/0/0 |
| dpt-full-reset | 5/8 | 2/8 | 0/1/1/1 | 0/0/1/0 |

## Matched case differences

- AD full versus AD static, bottle: full succeeds and static fails at `adjust_bottle:E1:requested22261030:actual22261030`; the result reverses at `adjust_bottle:E5:requested22265018:actual22265019`. Both finish5/8.
- AD full-reset gains the bottle E5 case above over same-weight AD full, finishing6/8 versus5/8.
- DPT static gains `adjust_bottle:E1:requested22261030:actual22261030` over DPT full, finishing6/8 versus5/8. These are separately fitted weights.
- DPT full and same-weight DPT full-reset have identical bottle success outcomes,5/8 each.
- DPT full-reset succeeds on `move_stapler_pad:E4:requested22264021:actual22264021` and `move_stapler_pad:E6:requested22266015:actual22266015`; DPT full fails both, giving2/8 versus0/8. All other stapler units finish0/8.
- Every condition’s eight actual seeds, initial raw-observation hashes and initial feature hashes exactly match its goal’s native case bank. The bottle E5 requested seed falls back to actual22265019 consistently across all nine conditions; stapler E5 remains22265018.

## Same-weight episode1 control

For all four AD/DPT × goal pairs, full and full-reset have identical saved feature, next-feature, sampled-noise and issued motor-target NPY bytes at **every episode1 decision**, with identical reward, termination, truncation and action-duration traces. E1 is a success for both AD bottle units and a failure for both units in the other three pairs. No E1 divergence was found; the two DPT-reset stapler successes occur later at E4/E6. No RGB members were read for this comparison.

| Pair | E1 decisions in each unit | First sampled-noise NPY SHA256 |
| --- | --- | --- |
| ad / adjust_bottle | 5 | `e98ad90fe352dc6476934f86a2fbafd37747d628d0cdf04fb0913e1ca644a67d` |
| ad / move_stapler_pad | 13 | `b650680e481c80b5815bd78281c1b1cd6b46c06cc112e27dcda26acb0cdfbb7f` |
| dpt / adjust_bottle | 13 | `e0df4917ddd044fd39c3853cada1ffaba75a11b266f24fac509508802361c877` |
| dpt / move_stapler_pad | 13 | `aa5b22dd0dbb17713135f942c433c823d4c66b21256cd43f08bc1ca516e00fa9` |

The initial model token is not stored as a separate blob. Its matching public feature bytes and the resulting first noise/motor proposal were inspected directly; the entire E1 trace matches. Typed-array digests are also retained in the detailed evidence.

## Identity and runtime checks

- Execution source `2fd2096906e52aabeaeb984f333fe1aaa6c4e196`, native backend `eacc64b4daf8f78aeb5e023a2c4d2a2a040eacfd2e662335c118334c2ab50ff3`.
- Plan file SHA `b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c`; canonical plan SHA `3e82a6df6001024cffaae92af26a2412fb11b260302504ec1ae4243d06b6b5f1`. Both were independently recomputed.
- All 18 completed-unit receipt hashes and result hashes match. The fixed order covers exactly nine conditions × two goals × eight episodes,144 goal episodes total; episode1/2/4/8 reports match the underlying outcomes.
- All five model/teacher checkpoint file hashes match the admitted first-fit100 bank. Frozen steering parameter digests match the five CPU admission identities; reset variants share the corresponding full checkpoint and parameter digest.
- All18 zero-action qualification receipts report native-noise parity max error0.0. 189 renderer receipt files passed SHA checks and bind rendering to allocated GPU PCI `0000:a3:00.0`.
- Loaded motor admission binds `721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41`. The pinned trial code re-hashes steering parameters and recomputes the motor fingerprint after all eight episodes, raising before result publication if either changed. All completed results therefore passed these guards; a separate literal post-motor hash field is not stored.
- This audit compared stored model parameter digests to admissions; it did not reconstruct models or independently recompute parameter digests from tensors.

## Physical cost and closed boundaries

The panel consumed **47,137 actual primitive control actions**, across 1,540 complete decisions. All 3,100 ledger events passed independent sequence/hash-chain replay and reproduce the final snapshot and summary exactly. Native bool valid-action-mask sums were inspected for every raw transition and match ledger durations; every unit’s corpus, result and cumulative ledger boundary agree. Pending and uncertain charges are both0.

The separate evaluation ledger has target cap57,600 and source_caps={};10,463 actions remain under that ceiling. All recorded actions are target evaluation and additional SGD updates are0. Qualification/reset steps issued zero policy actions. Internal reset physics ticks are outside this control-action unit.

Wrapper elapsed time is 2938.385s (48.97min); the recorded GPU-seconds value is a lower bound, not independently queried scheduler billing. The sum of raw native-step durations is 562.970s and excludes policy inference/setup overhead.

## Audit scope and evidence

The audit used only standard-library JSON, SHA, ZIP/NPY artifact inspection. It did not invoke model constructors, forwards, tests, CUDA, simulator work, ledger constructors/locks, or native process controls. Original outputs, code, coordination and index files were not changed. Full historical raw RGB/frame files were not rehashed. Reset pairing is checked through hash-bound stored reset/episode identities; features/noise/motor NPY members were read only for E1 same-weight comparisons.

- [audit-compact.json](audit-compact.json) — SHA256 `e06292c637c48287afb9be0a0c8b26cbac21683390155a9aba7073118f8ff738`.
- [audit.json](audit.json) — SHA256 `0845c894f86a4225320f7648f051f7369af369c04dda45d05adad3df58a9310c`.
- [episode1-controls.json](episode1-controls.json) — SHA256 `91f7a100057cfd825c3c32ecf614284c556ffb6b1ff37f81fe67415b76200e4a`.
