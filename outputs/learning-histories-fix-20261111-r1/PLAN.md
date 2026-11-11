# Learning histories: narrowing the gap to the enumeration reference

Not launched. Follows the [fifteen-fit grid](../learning-histories-20261111-r1/RESULT.md), where attempts, corrections and linked feedback stopped about 25–29 points short of the full-prior enumeration reference after six events. The [paired follow-up](../learning-histories-feedback-20261111-r1/RESULT.md) established causal reward use but a substantial fitting deficit even on training rules. This is a recipe screen, not a factorial attribution of that deficit.

The question is narrow: which change to the learner lets it accumulate evidence across scenes? Four of the five variants use the `corrections` condition. It is plain supervised in-context learning with no attempt policy, so a deficit there belongs to the model and recipe.

| Member | Condition | Change from the first grid |
| --- | --- | --- |
| `both` | corrections | `--dense --fourier 4` |
| `base-lr` | corrections | `--lr 1e-3` only |
| `both-lr` | corrections | `--dense --fourier 4 --lr 1e-3` |
| `wide` | corrections | `--dense --fourier 4 --lr 5e-4 --d 256 --layers 6 --heads 8` |
| `attempts-both-lr` | attempts | `--dense --fourier 4 --lr 1e-3` |

- `--dense` scores every scene of a session as a query on the events before it: eight labels per session instead of one, through a block-causal attention mask. It changes attention as well as loss density: event tokens cannot attend to query tokens, unlike the original bidirectional learner. One eighth of the objective is the first scene with no demonstration. An improvement cannot be attributed to supervision density alone.
- `--fourier 4` adds four sinusoidal octaves per attribute.
- Three seeds each, K=6, 20,000 updates, batch 256, 8,192 evaluation sessions per split, as before. The unchanged `corrections` fits of the first grid are the baseline.
- 15 one-GPU workers through `run.sh <commit> <identity> <qualification-identity>`. The first grid took 29 minutes at 0.09 s per update; dense passes hold 75 tokens instead of 40 and `wide` is larger. Launch only after CUDA qualification projects each fit below 80 minutes, leaving evaluation margin inside the 90-minute worker limit. Maximum grid allocation: 22.5 GPU-hours.
- All variants retain 20,000 updates. This does not test longer optimization. `wide` also changes learning rate, so its result cannot establish a capacity effect by itself.

## Earlier exploratory evidence

- One dense pass equals separate truncated passes for every condition, to 5e-7 in logits. Changing the answer of a scene leaves that scene's own query logits unchanged, so the mask does not leak labels.
- A partial `corrections` comparison, one seed, batch 256, stopped early for time. Training loss left the chance plateau (1.61) by step 1,500 with `--lr 1e-3` and by step 1,000 with `--dense --fourier 4 --lr 1e-3`; at the original rate neither the base learner (3,000 steps) nor `--dense --fourier 4` (1,500 steps) had left it. All that left it then sat near 1.05–1.1. Dense loss averages over eight positions and is not comparable with the single-query loss.

These are the CPU checks reported in the incoming plan, not qualification on the target runtime. That partial signal says the learning rate shortens the first plateau. It does not say any variant passes the second one, which is what this grid tests. All further model computation runs on allocated Linux resources.

## Required CUDA qualification

Run `qualify.sh <commit> <fresh identity>` on one allocated Baige GPU with a 30-minute limit (0.5 GPU-hours). The script records the exact commit, storage/write check and in-job GPU probe, then checks:

- Dense shared-pass logits and gradients against separate causal prefixes for all conditions and representative architectures; current/future events and other queries must not leak into a query.
- Bitwise checkpoint continuation for the legacy schema and dense learner, including optimizer, scheduler and RNG state.
- Exact replay of the original attempts seed-0 curves from its preserved checkpoint. Older checkpoints omit the new architecture arguments and must load with their historical defaults.
- Batch-256 capacity and 200-update timings for base, dense and wide recipes; peak allocated GPU memory and projected 20,000-update duration.

The grid requires the passed receipt from that same commit. A failed check is diagnosed before a fresh qualification identity is used. Do not increase the allocation or change a live contract silently.

## Evaluation and budget

Use the original evaluation RNG base 10000 only as a development panel for comparison with the first grid. After selecting a recipe, confirm it on fresh sessions with `--eval-seed 90000`, matched to a re-evaluation of the corresponding original checkpoints and enumeration references. Selection-panel gains are exploratory until that confirmation. Reserve at most 0.25 GPU-hours for confirmation; adjust the bounded workload after measured evaluation timing.

Prior toy work used 7.5303 scheduler GPU-hours (7.7678 conservative bound). This screen reserves at most 23.25 additional GPU-hours including qualification and confirmation. The separate LIBERO-Gen motor pilot is capped at 5 GPU-hours, giving a combined conservative bound of 36.0178 GPU-hours, below the approved 50-hour threshold.

## Reading the result

Use held-out `corrections` accuracy against the reference on the same histories.

| Outcome | Next step |
| --- | --- |
| A variant is within 5 points at every k | Confirm on the fresh panel, check `attempts-both-lr` the same way, then propose the five-condition comparison with that recipe. |
| Best variant closes part of the gap and `wide` is best | Run a matched-learning-rate size comparison before attributing the gain to capacity. |
| No variant improves on 67.5% at k=6 | These recipes failed; optimization remains possible. Inspect per-rule errors, one- versus two-attribute rules and training curves before choosing a longer fit or a different learner. |
