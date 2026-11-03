# What the completed OpenDrawer rubric trials establish

There is no qualified OpenDrawer Flora first-success reward yet. The tests
support a narrower diagnosis than either “more detail fixes it” or “rubrics
cannot work.” Keep the simulator definition explicit, and stop selecting
successive rewrites on the same few errors without a new diagnostic question.

## The task definition that must survive any change

- The requested drawer is selected using robot-relative left/right at reset
  and the native TOP_DRAWER selector. That selector is a 0.7–0.9 m height
  band. All ten audited layouts put the selected front at z=0.785 with no
  drawer above it; uppermost geometry is not proven for every possible scene.
- Native success is a slide displacement of at least 0.95×0.55 of the fixture
  depth: 52.25%, not 95% of the mechanical range. Maximum extension, releasing
  the handle, and opening every drawer are not required. Moving a cabinet door
  or another drawer does not satisfy the target predicate.
- First success is the first control satisfying that predicate. Later closing
  does not undo it. Cases 64 and 216 succeed and then close, so their terminal
  failure labels cannot be reused as never-success labels.
- The two-Score-5 checks 50 controls apart are a proposed detector rule with
  an allowed delay, not an extra native persistence requirement. No score-token
  probability has been independently calibrated as success probability.

These definitions follow the retained source/replay evidence in
[the native diagnosis](../robocasa-opendrawer-rubric-diagnosis-20261102-r1/RESULT.md),
[the added failure replays](../robocasa-flora-drawer-extra-failures-20261103-r1/RESULT.md),
and [the geometry audit](../robocasa-flora-first-success-rubric-20261103-r2/TARGET-AUDIT.json).

## Controlled results that change the next action

| Intervention | Evidence | Consequence |
| --- | --- | --- |
| Direct 0/1 output | Revision 1.1 accepts 9/10 sampled terminal-failure drawer videos, versus 7/10 with graded output. | Binary output alone did not repair the terminal metric. The later first-success relabeling means this is not a first-success false-positive rate. |
| Separate target and depth clauses | On six videos / 35 prefixes, dropping the target block recovers 128 but falsely confirms wrong-drawer 112. Depth-only produces score 4 everywhere after success, and also on 112. | The target block exposes a recognition tradeoff; accepting score 4 does not solve it. |
| Delete only the stack sentence, or separate location and exclusion | 128 still misses or becomes inconsistent; wrong-drawer acceptance returns under several shorter variants. | Neither that one sentence nor the negative clause is established as the sole cause. |
| Last 50 or 100 controls with fixed rubric | Neither window confirms any of the five successful cases in the six-video panel. | Recent chunks, with their changed context and sampling, did not repair recognition. |
| Short exact-threshold task descriptions | Both give 0/27 false full scores and 0/18 true post-success full scores on the expanded panel. | Zero false scores obtained by suppressing all successes is not a repair. |
| Explicit system rubric and grade anchors | Anchors raise true post-success full scores 11→15/18, but false full scores 18→19/27. Both confirm all four never-success cases. | More explicit score meanings improve sensitivity and also admit known failures. |
| Exact boundary plus separate identity hints | In the latest 26-prefix screen, the exact base gives 0 false full scores but no confirmed successes; target tracking gives 1 false full score and only 1/6 confirmed successes; the shallow-drawer hint gives 2 false full scores and 1/6 confirmed successes. | None passes. 128 remains score 1 at controls 350, 400 and 450 under every fresh variant. |

The table summarizes different matched comparisons, not a common leaderboard.
Their banks, native label definitions, endpoints, and primary rules differ.
All are reused, error-enriched development evidence. No independent population
precision, online first-success qualification, or improving PPO reward follows.

Primary run evidence:
[binary/graded](../robocasa-flora-rubric-failure-sample-20261102-r1/RESULT.md),
[clause ablation](../robocasa-flora-drawer-clause-ablation-20261103-r1/RESULT.md),
[target wording](../robocasa-flora-drawer-target-wording-20261103-r1/RESULT.md),
[location/exclusion](../robocasa-flora-drawer-location-exclusion-20261103-r1/RESULT.md),
[chunks](../robocasa-flora-drawer-chunk-rubric-20261103-r1/RESULT.md),
[exact threshold](../robocasa-flora-drawer-exact-threshold-20261103-r1/RESULT.md),
[grade anchors](../robocasa-flora-drawer-grade-rubric-20261103-r1/RESULT.md),
[target hints](../robocasa-flora-drawer-target-hints-20261103-r1/RESULT.md).

## The input question being tested now

The actual RGB reads in the latest scoring run match CPU references exactly.
128@450 includes ten post-native opening frames. The passing r3 audit now
reconstructs all 256 frames byte for byte from the official CPU processor,
including after the bfloat16 input conversion. Its six exported artifact
hashes and all source-frame hashes verify, and the actual 0.5 normalizer
matches the saved config. This reconstructed preprocessing stage preserves
the sampled pixels. It cannot prove semantic retention through learned visual
embeddings, runtime cache correctness, or equality to earlier GPU tensors
that were never saved. See [the passing result](RESULT.md).

The next scoring decision must follow that result. Do not launch another
combined wording change merely because the previous one failed. A new
diagnostic must name the ambiguity it separates, keep all task criteria
explicit, include wrong-target and insufficient-opening failures plus missed
and recognized successes, and predeclare timing and rejection rules. Any
candidate that passes development still needs new retained cases that were
not used for wording selection before production promotion. Keep the complete
45-prefix development panel as the next validation stage, not a population
test or independent holdout. No new GPU scoring is authorized by this review.
