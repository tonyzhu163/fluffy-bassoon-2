# Earlier attempts help select a retry cadence on LIBERO development tasks

A frozen reader used the first attempt's public motion to select a cadence for
frozen OpenPI π0.5 on a fresh reset. It succeeded on **20/32** retries across
eight LIBERO-Pro2.0 development tasks. Fixed default execution scored **18/32**
and fixed pacing scored **14/32**. The matched query branches give a ceiling
of **21/32** for this menu. [Protocol](PLAN.md) · [Per-trial results](paired01-summary.json)

| Retry decision | BASE | S20 delay | Total |
| --- | ---: | ---: | ---: |
| Clear prior experience, always default | 16/16 | 2/16 | 18/32 |
| Clear prior experience, always pace | 10/16 | 4/16 | 14/32 |
| Keep prior experience and infer timing | 16/16 | 4/16 | 20/32 |
| Select using true timing | 16/16 | 4/16 | 20/32 |
| Either measured mode succeeds | 16/16 | 5/16 | 21/32 |
| Both measured modes succeed | 10/16 | 1/16 | 11/32 |

The first attempt used default execution and succeeded on 16/16 BASE and 2/16
S20 starts. The reader used only the first 80 issued translations and observed
end-effector positions. The offset estimates were 0–1 under BASE and 8–15
under S20. The demonstration-calibrated rule identified all 32 timing settings
without abstaining or reading rewards. Each decision was written before the
selected query ran. The other query mode used the identical initial simulator
state, public observation tensors, action proposal, paired policy-noise stream
and physical control budget.

Compared with fixed default execution, the rule gained three retries and lost
one, all under S20. Pacing recovered both stove placements (Goal 1) and one
Spatial 1 placement. Switching to pacing lost one successful middle-drawer
opening (Goal 0). Compared with always pacing, the rule gained six retries and
lost none, with all six gains under BASE. These are descriptive counts from
16 task/reset-pair blocks crossed with timing.

For decision pretraining, BASE produced six pairs with a unique successful
mode and ten with two successful modes. S20 produced four pairs with a unique
successful mode, one with two, and **11 with neither**. The reader already
reached the success of timing-aware mode selection on this panel. A richer
reader could gain at most one additional delayed success from these measured
actions. Improving the available continuations matters more here than decoding
timing more accurately. These development comparisons used a fixed motion rule
with no learned policy or reward-dependent decision. The official test tasks
remain reserved.

Baige development RTX 4090, `paired01`, commit `c45aa965`, exit 0 at 10:20:48
Shanghai on Nov 19. The 32 supports and 64 queries used 0.475 GPU-hours
including qualification and startup. We read all 96 archives back to verify
the reset pairing, initial proposals, command counts, waits and native
outcomes. Raw records are on Baige at
`/mowenjie/fluffy-bassoon-2/outputs/libero-retry-headroom-20261119-r1/paired01/`.
[Worker](pilot.py) · [Readout and checks](summarize.py)
