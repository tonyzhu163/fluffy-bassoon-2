# Frozen RLinf pi0.5 LIBERO-130 motor

The requested motor is
[`RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT),
revision `6222623f635769bfc73c9472e29fab9b7fd8e027`. Its inference weights
are 7,473,091,464 bytes, SHA256
`4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683`.
Its own `physical-intelligence/libero/norm_stats.json` is required, SHA256
`dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`.
Optimizer and pickle metadata are unnecessary for frozen inference.

The release has no model card. RLinf's
[RPent documentation](https://rpent.readthedocs.io/en/latest/rst_source/leaderboard/performance.html)
identifies this named checkpoint as supervised fine-tuned on LIBERO-130 and
kept frozen for LIBERO-PRO. This provenance does not turn any original LIBERO
task into a task unseen by the VLA. Controller training on LIBERO and testing
on external Plus/Pro shifts must report that distinction.

`vla_rlinf_libero.FrozenRLinfPi05Libero` uses pinned RLinf
`e1a82ebf0c1af69da6ab9b1519f5f3692fe33566`, its installed `rlinf-openpi 0.1.1`,
and the [native pi05_libero configuration](https://github.com/RLinf/RLinf/blob/e1a82ebf0c1af69da6ab9b1519f5f3692fe33566/rlinf/models/embodiment/openpi/dataconfig/__init__.py).
The interface predicts ten actions, uses 32 initial-noise coordinates and ten
Euler steps, and decodes seven LIBERO control coordinates. Explicit initial
noise is passed to native `Pi0Eval.predict_action_batch(mode="eval")` before
denoising; no RL exploration, critic or steering head is initialized.

The backend verifies weight/normalization bytes and pinned motor source files.
Native checkpoint key conversion is retained, with strict complete loading.
All motor parameters are frozen. Public inputs are two RGB views, eight robot
state coordinates and the instruction. Native preprocessing and the checkpoint's
normalization are used. Optional 4,104-dimensional features pool native final
visual/language prefix states separately and append public robot state; their
contract names this backend and checkpoint, independently of earlier OpenPI runs.

All construction and inference require an allocated Linux GPU. Existing
environments and checkpoints are read in place. Real simulator qualification,
same-noise parity and actual GPU bindings are required in each new runtime.
