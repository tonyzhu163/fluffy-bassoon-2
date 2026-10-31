# LIBERO-Pro development adaptation POC

The frozen motor is the released RLinf pi0.5 SFT checkpoint trained on all 130
standard LIBERO tasks. This is a development test of native performance and
ordinary DSRL adaptation under official Pro shifts, before training or evaluating
a frozen history controller. It is not a final leaderboard score or a semantic
task holdout claim.

Before any Pro outcomes are collected, the two cases are fixed as follows:

| Case | Official data folder | Initial state | Trial seed |
| --- | --- | --- | --- |
| Position swap | `libero_object_swap` | 0 | 0 |
| Object appearance | `libero_object_object` | 0 | 1000 |

Both derive from original `libero_object:1`, cream cheese into basket. Its full
goal and typed object binding match the TRAIN family of `libero_90:47` in the
frozen source split. The appearance shift replaces the target and basket with
official red variants and inherits the original base's lineage. This is not a
variant of task 57, whose goal uses a tray.

Official source is pinned to `Zxy-MLlab/LIBERO-PRO` commit
`eafdb809426b13153aa1e4c42d6601844217dfec`. The four BDDL/init files come from
`zhouxueyang/LIBERO-Pro` revision
`c86fc3b8293185a6f373677018ff3e37f8391602`. Downloaded bytes were checked against
the dataset's Git blob hashes and recorded with SHA256; receipt SHA256 is
`98a5cae87d4939303f451c955b6dfc478f720ee0fd83fa611b5a771a1478fefe`.
The clean source includes 864 asset files, 644,001,017 bytes, with no LFS pointers;
asset-manifest SHA256 is
`90a08e2df84608921b0f072bce6d902e1657cafbc9b847c9947a2a841f7f3acb`.

The episode horizon is 280, matching the Pro README's object suite protocol.
Five `[0,0,0,0,0,0,0]` settling actions follow the exact pinned
`benchmark_scripts/render_single_task.py` lines 22 and 66. That source is the
official rendering utility; it does not supply the complete leaderboard model
evaluation recipe. Settling steps are counted separately from controlled robot
interactions. Each physical episode constructs a fresh simulator after seeding
NumPy/Python, sets the supplied initialization, and records observation/state
hashes. Only state zero is admitted by the worker.

The shared shifted driver evaluates native episodes and ordinary online DSRL,
with four attempts per declared case, bounded 32D steering, chunk 10, SAC UTD20,
batch32 and warmup32 chunks. All methods use the same case state, seed and
280-step cap. Native noise and action transformation follow the exact released
RLinf runtime. The pi0.5 weights stay frozen. Complete source recoveries and raw
episode evidence are retained. No outcome may trigger extra cases or a larger
collection without a new declared plan.

Runtime staging uses the existing simulator Python at
`/mowenjie/envs/wam-libero-plus-py310/bin/python` with the new pinned Pro source
first on its Python path. The model uses the existing RLinf Python and its
installed OpenPI package. No shared environment or active run is modified.
