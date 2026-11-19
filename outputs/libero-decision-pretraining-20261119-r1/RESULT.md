# Decision pretraining on a four-mode LIBERO retry menu

The first trained categorical Transformer changed modes when given an
earlier attempt. It reached **20/32** native development successes in each
of three fits. Clearing the same reader's history gave **20, 20 and 19/32**.
The simple lag rule reached **21/32**. The new wait modes supplied one
additional successful fitting query and zero additional successful
development queries beyond default and fixed pacing. [Plan](PLAN.md) ·
[Trial-level readout](reader01-summary.json)

## Same-state action comparison

Each panel used eight tasks, two support/query reset pairs and two timing
conditions. Every trial ran one support and four queries. The four query
branches shared the initial simulator fingerprint, public observations,
first π0.5 proposal, policy noise schedule and native horizon. The measured
endpoints are successes at native termination.

| Execution choice | Fitting BASE | Fitting S20 | Development BASE | Development S20 | Development total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default, five controls per inference | 14/16 | 2/16 | 16/16 | 2/16 | 18/32 |
| Ten controls and fourteen waits | 6/16 | 3/16 | 8/16 | 5/16 | 13/32 |
| Ten controls and seven waits | 12/16 | 3/16 | 14/16 | 4/16 | 18/32 |
| Seven waits, fourteen on gripper transitions and the opening burst | 12/16 | 2/16 | 15/16 | 4/16 | 19/32 |
| Fitting-selected timing rule from public lag | 14/16 | 3/16 | 16/16 | 5/16 | 21/32 |
| Any measured mode succeeds | 15/16 | 4/16 | 16/16 | 6/16 | 22/32 |

The fitting-selected fixed strategy was default. Both shorter-wait modes
improved nominal performance relative to always pacing. They did not rescue
any of the ten delayed development queries that failed both older modes.
On fitting, one delayed plate push (Goal 5) became feasible. Only 19/32
fitting queries supplied a positive categorical target, including 4/16 under
delay. Twenty-nine fitting queries either had multiple successful modes or
all four modes failed. Consequently only three gave a unique successful
target.

The rerun matched all 32 archived development reset identities. Three paced
outcomes differed from the old worker. The two baseline modes in the table
come from this rerun with the same runtime as the two new ones. Its higher
two-mode ceiling should not be credited to the added waits.

## Frozen reader on development tasks

The reader used five-control tokens from the first 80 support controls and
a 2,056-dimensional frozen query feature. It used the same two-layer
128-wide architecture in each fit. Targets were uniform over successful
branches. All training consumed fitting tasks only. The evaluation selected
from reset-matched, previously executed physical branches.

| Fitting seed | Retained | Cleared | Opposite-timing support | Separately fitted current-only | Retained wins and losses against cleared |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 20/32 | 20/32 | 20/32 | 19/32 | 1 and 1 |
| 42 | 20/32 | 20/32 | 20/32 | 19/32 | 1 and 1 |
| 43 | 20/32 | 19/32 | 19/32 | 19/32 | 2 and 1 |

Retaining support changed the chosen mode on 16, 19 and 20 of 32 queries,
respectively. Substituting the other timing's support changed 8, 16 and 16.
Those changes yielded little success advantage. All retained fits reached
16/16 nominal and 4/16 delayed. Against the lag rule each gained one
paired success and lost two. The three seeds share the same 16 development
task/reset pairs crossed with timing.

Keep the reader as a working baseline. Scaling this recipe would address a
small residual gap on the measured menu. To increase the opportunity to
learn, qualify corrections that create successful alternatives on starts
where pacing fails. Compare them with always applying the correction before
attributing gains to memory. This run measured second-attempt decisions at
the reset. The shared-current-prefix diagnostic, fresh live learner
comparison and official held-out panel remain separate evaluation steps.

## Execution

Baige development RTX 4090, `menu02`, pinned to `b90ee5a0`, exit 0 at
11:49:14 Shanghai on Nov 19. The 320 simulator episodes used 0.799 GPU-hours,
including the probe and startup. The CPU readout verified all branch archives
and saved six checkpoints with optimizer state. All six reproduced the saved
development choices when reloaded. A first queued launcher was
withdrawn before GPU acquisition so the fitting and development noise seeds
could be made disjoint. Raw outputs and checkpoints are on Baige under
`/mowenjie/fluffy-bassoon-2/outputs/libero-decision-pretraining-20261119-r1/`.
[Collector](collect.py) · [Reader and audit](reader.py)
