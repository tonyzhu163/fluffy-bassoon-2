# Native qualification of Stove replacement candidates

Both candidates completed 50 fresh cases at 600 controls with the unchanged
released Human300 pi0.5, H50 commands, ten flow steps and three cameras.
Baige `job-ltf38050md7y` succeeded and released its two GPUs at
2026-11-04 13:07:44 Shanghai. Execution source: `a28f8405`.

| Candidate | First success at 600 | Wilson 95% | Success at end | First success by native horizon |
| --- | ---: | ---: | ---: | ---: |
| TurnOnSinkFaucet |18/50 (36%)|24.1–49.9%|18/50 (36%)|18/50 at 600|
| TurnOnElectricKettle |32/50 (64%)|50.1–75.9%|29/50 (58%)|32/50 at 450|

The rule declared before these outcomes selects **TurnOnSinkFaucet**: prefer
Faucet if first success is 3–35/50, otherwise use Kettle if it meets that range,
otherwise neither. Both qualify under that original 5–70% point-estimate band.
This is native headroom qualification; improving source learning and Flora
agreement remain unqualified. No new PPO training or student fitting was run.

The human subsequently requested a fresh justification of the whole task
portfolio and suggested an approximately 30–60% preferred baseline range.
That review does not retroactively change this run's rule or outcomes:
[18-task selection and Faucet/Kettle review](../robocasa-atomic-portfolio-review-20261104-r1/REVIEW.md).
Faucet's proposed move from the prior OOD list to ID leaves an OOD slot to
resolve before a new source/meta-learning contract is adopted.

Independent Linux CPU verification passed for all 100 exact case identities,
1,200 native chunk/noise schedules, zero residual outputs and exact base/executed
chunk equality. All 300 three-view videos match their recorded hashes and
fully decode to 601 frames each (180,300 frames total). Physical accounting
is 60,000 controls with zero unresolved intents and zero SGD. The collector's
online replay flag covers the first case in each worker; the independent
post-run audit checks every saved action chunk.

Conservative creation-to-release cost is **1,724 GPU-seconds (0.479 GPU-hours)**,
within the 6,000 GPU-second cap. The process clock alone records 1,654 GPU-s.
All preparation/audit repairs were on allocated Linux CPUs and spent no GPU time.
The first preparation fixed a strict floating-point Wilson endpoint assertion;
the audit then bound native instruction text and the collector's first-case
online-flag semantics before checking all actions independently.

- [Final independent audit](FINAL-AUDIT.json), [decoded video manifest](VIDEO-AUDIT.json).
- [Raw aggregate](RESULT.json), [per-case results](CASE-RESULTS.json), [source and GPU allocation](PROFILE.json).
- [Terminal accounting](TERMINAL-ACCOUNTING.json), [process clock](PROCESS-CLOCK.json), [Linux qualification](CPU-QUALIFICATION.json).

Raw videos and episode tensors remain on project PFS in
`/mowenjie/fluffy-bassoon-2/outputs/robocasa-buttons-replacement-qualification-20261104-r1/gpu-r1`.
