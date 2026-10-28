# Tracker capacity gate results

Checked 2026-10-28 22:25:09 Shanghai. Baige `job-o16q3eimo8bq`, exact source `c2c47152303ccfc1ac469fd3e1ad1e1344201e42`. The driver finished in 826.63 seconds, exit code 0, without a bounded stop. Allocated Linux qualification passed 151 tests in 20.02 seconds.

The capacity gate is **unresolved**: the bounded privileged tracker scored 4/6 success-once against zero 3/6 and external oracle 4/6, but failed the declared requirement for at least one success at each perturbed angle. Crossed-history evaluation was not run; PPO was not launched.

| Hidden angle | Bounded tracker | Zero | External oracle |
| --- | ---: | ---: | ---: |
| −45° | 0/2 | 0/2 | 1/2 |
| 0° | 2/2 | 2/2 | 2/2 |
| +45° | 2/2 | 1/2 | 1/2 |

Gain 1.0 was selected at both perturbed angles by the declared training-scene criterion. This is a privileged analytic controller through the real four-Euler, seven-active-coordinate residual path bounded at ±1; it is not learned CTM competence or an ICRL result. Six paired episodes per condition are a diagnostic screen.

The independent audit passed **297/297 checks** over all 18 trajectory files and all 16 chunks in each file. It recomputed the hashes of actual float32 decoded/prepared/issued command arrays, state/feedback snapshots and original/prepared input trees; verified all file hashes, noise-generator continuity and unchanged non-XY commands; recomputed success-once from all 80 simulator-step outcomes; and reconciled gain selection, the failed gate rule and every cost counter. No model was executed during this audit and no large trajectory or checkpoint was copied.

Every same-policy zero reference and its repeated rollout were exactly equal across their full saved chunk rows, including actual commands, native input digests, noise, feedback and step outcomes, at all three angles. This establishes stability for these three repeated pairs. It does **not** explain or globally resolve the earlier cross-method zero discrepancy in the support/probe run.

| Angle | Maximum absolute residual | Mean clipped fraction | Mean executed-target model MSE |
| --- | ---: | ---: | ---: |
| −45° | 1.0 | 3.2924% | 0.00557019 |
| 0° | 0.0 | 0.0% | 0.0 |
| +45° | 1.0 | 2.9855% | 0.00412260 |

These are recorded native-sampler diagnostics, averaged over the 16 chunks. Target MSE is in model-action coordinates over the executed five-step prefix. They do not identify why an episode failed. The audit verified reported residual means stay within the declared bound; full per-scalar residual tensors were not saved.

Actual cost independently recomputed from all completed trajectories and extra-query records: **2,880 simulator steps, 576 policy chunks, 928 lane-weighted native prefix queries, and 3,712 native suffix queries**. Extra full motor calls comprise 96 external-label calls and 256 tracker-target calls. The frozen motor fingerprint is unchanged before/after: `ed862a89a39052dad591d6542959ef0a72866b2c85eac2c84767bf55f52205dc`.

The final summary SHA256 is `16836f57e40f8c28583a3bfd4b9c6e0f45b05028dcd9a520572e4477faffa470`. The independently audited final checkpoint SHA256 reported by the separate metadata reviewer is `48fd40db4f479653b3e4dac1502e6f6081c9b27e828ddfea71ab4a91864da79a`; that checkpoint audit is maintained with the integration evidence. All 18 actual trajectory hashes, per-file checks, residual summaries and terminal receipts are in [audit/terminal-audit.json](audit/terminal-audit.json). The read-only stdlib audit is [audit/read_saved_trajectories.py](audit/read_saved_trajectories.py).
