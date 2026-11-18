# Preparation status, 2026-11-02 Asia/Shanghai

No GPU fit, simulator evaluation, or scheduler submission was launched.
The human questioned the draft's self-imposed resource caps before launch;
the draft remains provisional. The unsubmitted allocation form was closed.

Allocated Baige development CPUs, CUDA hidden: 27 core tests passed in7.53s
at `ccb9f16c4` (residual history, native binding, history policy, decoder).
The initial exporter correctly stopped on Python tuple versus JSON list
contract representation; normalized JSON equality was added at `c129fd14c`.
The complete offline export then passed, changing only DPT target arrays.
No source records or checkpoints were modified.

Canonical derived data:
`/mowenjie/fluffy-bassoon-2/outputs/opendrawer-ppo-ad-dpt-20261102-r1/data/`.
[Assembly](assembly.json) records all dataset hashes:56 TRAIN streams,
336 episodes/5040 positions;8 VAL streams,48 episodes/720 positions.
The same pinned final TRAIN PPO actor labels both splits. Student validation
states were seen by source PPO and are not independent teacher holdouts.

Actual source successes per64-episode behavior version:
35,37,26,32,34,33 (197/384 total). This does not demonstrate improving
training success; retain the full learning history and call DPT labels
candidate-teacher supervision. Earlier fixed-eight ID improvement remains
preliminary. New held-out physical outcomes are needed for student comparison.

The local main checkout remains unchanged, including its pre-existing dirty
coordination edit and divergent commits. Work is isolated in the attached
`opendrawer-ad-dpt` detached worktree. No branch was published.
