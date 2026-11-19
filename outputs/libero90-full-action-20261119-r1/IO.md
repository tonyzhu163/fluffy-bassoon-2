# Full-action LIBERO-90 rollout interface, version 1

The manifest is the sister run's `../libero90-meta-retry-20261119-r1/cases.json`. Paths below are relative to the episode root given to the runner. A published trial contains `trial.json` with its exact `manifest_row`, source pin, reset fingerprints, episode costs and native outcomes. Writers publish the entire directory by atomic rename. Readers consume only published trials.

    fitting/task00-r00/{support,query0,query1}/
    development/taskNN-rNN/{support,query0,query1}/
    live/{development,test}/taskNN-rNN/{support,base,matched,meta_intact,meta_cleared,imitation_intact,imitation_cleared}/

Every episode has `public.npz` and `result.json`. Load archives with `allow_pickle=False`. The public tape has chronological issued actions `issued[T,7]`, post-control wrist `positions[T,3]`, pre-first-control `initial_position[3]`, post-control jaw `gripper[T,2]` and native `rewards[T]`. The scalar `controls` is T. The scalar boolean `terminal` is native terminal success. A horizon failure retains all 400 controls, has terminal false and is marked truncated in result.json. The JSON fields `controls`, `score` (sum of native rewards), `success`, `truncated`, `policy_calls`, `feature_calls` and timings describe the completed episode. Settling controls are separate.

Each query also has `decisions.npz`. Its arrays are finite float32 `state[J,8]` (position, axis-angle, two jaw positions), `embedding[J,D]` (pooled frozen visual/language prefix, expected D=2048), `proposals[J,K,10,7]`, `scores[J,K]` (before execution), integer `index[J]`, `offset[J]` (zero-based physical control count at decision) and `length[J]` (executed prefix length, 1–5). Concatenate state then embedding to recover the frozen query feature. K=2 throughout except for live `base`, where K=1. Both collection branches choose by independent fair coins, with prospective constant zero scores. The native result labels only the chosen trajectory. All failed episodes are retained.

## Frozen reader import

The collector starts a persistent CPU-only Python subprocess in the reader environment. It imports the sibling `model.py`. That module exports this interface for each frozen recipe.

    scorer = load_scorer(path: pathlib.Path, recipe: str)
    scores = scorer(support, state, embedding, proposals)

`path` is a single self-contained portable weights file for the recipe, including ensemble and normalization parameters. Its SHA-256 is recorded. `recipe` is exactly `meta` or `imitation`. `support` is a dictionary of the seven numeric public tape arrays named above, or Python `None` for cleared history. The reader constructs its history including the episode boundary from this tape. `state` is float32 [8], `embedding` float32 [D] and `proposals` float32 [2,10,7]. The callable returns two finite scores in proposal order as a NumPy-compatible array, averaging logits across frozen seeds before returning. Calls are stateless with respect to query history. All tensor/device conversion belongs in the reader. The collector executes the first argmax. It supplies no query rewards, earlier query decisions, timing, IDs, seeds, sibling outcomes or evaluator metadata.

Across live policies, proposal noise is shared by trial, replan ordinal and candidate index. Each policy resets the simulator and visits its own observations. Base executes one draw. Matched computes both and executes candidate zero. For live matched and learned queries the collector also makes one feature call per decision. Base saves the same feature for analysis and its cost is recorded separately from the one native action inference. Support uses one native draw per five controls.
