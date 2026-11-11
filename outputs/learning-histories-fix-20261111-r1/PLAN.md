# Learning histories: closing the gap to the optimal reader

Not launched. Follows the [fifteen-fit grid](../learning-histories-20261111-r1/RESULT.md), where every condition stopped about 27–29 points short of the enumeration reference after six events, on training and held-out rules alike, with training loss flat near 1.0 from roughly step 8,000.

The question is narrow: which change to the learner lets it accumulate evidence across scenes? Four of the five variants use the `corrections` condition. It is plain supervised in-context learning with no attempt policy, so a deficit there belongs to the model and recipe.

| Member | Condition | Change from the first grid |
| --- | --- | --- |
| `both` | corrections | `--dense --fourier 4` |
| `base-lr` | corrections | `--lr 1e-3` only |
| `both-lr` | corrections | `--dense --fourier 4 --lr 1e-3` |
| `wide` | corrections | `--dense --fourier 4 --lr 5e-4 --d 256 --layers 6 --heads 8` |
| `attempts-both-lr` | attempts | `--dense --fourier 4 --lr 1e-3` |

- `--dense` scores every scene of a session as a query on the events before it: eight labels per session instead of one, through a block-causal attention mask.
- `--fourier 4` adds four sinusoidal octaves per attribute.
- Three seeds each, K=6, 20,000 updates, batch 256, 8,192 evaluation sessions per split, as before. The unchanged `corrections` fits of the first grid are the baseline.
- 15 one-GPU workers through `run.sh <commit> <identity>`. The first grid took 29 minutes at 0.09 s per update; dense passes hold 75 tokens instead of 40 and `wide` is larger, so budget 90 minutes per worker, 22.5 GPU-hours at most.

## Checked before launch (macOS, CPU)

- One dense pass equals separate truncated passes for every condition, to 5e-7 in logits. Changing the answer of a scene leaves that scene's own query logits unchanged, so the mask does not leak labels.
- A partial `corrections` comparison, one seed, batch 256, stopped early for time. Training loss left the chance plateau (1.61) by step 1,500 with `--lr 1e-3` and by step 1,000 with `--dense --fourier 4 --lr 1e-3`; at the original rate neither the base learner (3,000 steps) nor `--dense --fourier 4` (1,500 steps) had left it. All that left it then sat near 1.05–1.1. Dense loss averages over eight positions and is not comparable with the single-query loss.

That partial signal says the learning rate shortens the first plateau. It does not say any variant passes the second one, which is what this grid tests.

## Reading the result

Use held-out `corrections` accuracy against the reference on the same histories.

| Outcome | Next step |
| --- | --- |
| A variant is within 5 points at every k | Adopt it, check `attempts-both-lr` the same way, then rerun the five conditions with it. |
| Best variant closes part of the gap and `wide` is best | Capacity-limited: scale width and depth before anything else. |
| No variant improves on 67.5% at k=6 | The recipe is not the limit. Inspect what the learner computes (per-rule accuracy, one- versus two-attribute rules) before more fits. |
