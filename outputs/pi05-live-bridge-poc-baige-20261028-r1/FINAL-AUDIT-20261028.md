# Capped native PPO final audit

Baige `job-61p1rncjnr1x`, immutable source
`c3e9cca68b93b6e777cbe5314e130bcaf85c8e14`. Both workers exited 0 after
the existing driver cap. Both retained complete iteration-50 checkpoints;
the target was 100 updates, so this is a **partial 50/100 run**, not a
completed training schedule. No extension is admitted by this result.

Each worker collected 32,000 training simulator steps (6,400 action chunks),
performed 200 optimizer steps and 25,600 chunk replays; evaluation added
5,760 simulator steps each. Frozen motor fingerprints are unchanged.

| Learner | Normal | Memory wiped | Reward blind |
| --- | --- | --- | --- |
| CTM | 6/8 | 4/8 | 5/8 |
| Transformer-XL | 4/8 | 5/8 | 6/8 |

Each condition has only two shared-history K4 trials. These descriptive
fractions do not establish reliable ICRL, a reward effect or an architecture
ranking. The subsequent fresh-probe run uses pre-pinned iteration-42 inputs;
its active contract was not silently replaced by these later snapshots.

Saved small final receipts are in [final-evidence](final-evidence/); full
checkpoints and RNG/optimizer/sampler recovery state remain on Baige.
CTM checkpoint SHA256:
`f05cc50e934a00b307bf6fed172de77b7db1a02d64aa671d923cd36fd0494f29`.
XL checkpoint SHA256:
`207a29ee183120b8a6316f1aa1ec1aeda4dd1d387b63a33705db50fc6ff95274`.
