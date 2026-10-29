# RoboTwin setup for VLRLA-style adaptation

RoboTwin 2.0 is the primary candidate for a bimanual, multi-episode comparison;
LIBERO can qualify shared plumbing. [VLRLA v1](https://arxiv.org/html/2610.19517v1)
is the reference: section 5 supplies the six seen/six unseen task split,
Appendix A.1 the bridge, and A.2.1 the PPO history collector. Machine-readable
settings are in [configs/robotwin_vlrla.json](../configs/robotwin_vlrla.json).

## Runtime preparation

Use the RLinf-compatible RoboTwin branch, following the
[RLinf guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html).
The existing qualified dependency pins are RLinf
`c605b9d55ed93a180119b099e6f836abff293b08` and RoboTwin
`dca9ec682688821c42944eacd7f5dd9bfb15396f`. Preserve the sources, assets
and checkpoint files that earlier runs cite.

`scripts/setup/robotwin_vlrla.sh` creates a fresh Python environment with uv.
It reuses the shared runtime read-only through a recorded `.pth` and installs
MPLib 0.2.1 and Open3D 0.19.0 locally, preserving NumPy 1.26.4. Its seven
arguments are uv, base Python, new environment, new output directory, RLinf
source, RoboTwin source and model directory. Run on allocated Linux resources.
It verifies source pins, write access, dependency versions, all twelve task
imports, required asset roots and normalization availability. It records
before/after shared package versions. No simulator or model is instantiated.

CPU imports are not GPU qualification. In a scheduler allocation, run the
allocation-local probe and a real episode; verify renderer-device binding,
three nonblank camera views, finite [50,14] action chunks, unchanged motor
weights and advancing simulator steps. Resolve or explicitly qualify the
earlier SAPIEN denoiser warnings. Keep complete receipt hashes and source pins.

The cached `RLinf-Pi05-RoboTwin-SFT-adjust_bottle` checkpoint, revision
`fa8df6ed103db0f5549c122f3a17c00ba6426c98`, can qualify the runtime.
`adjust_bottle` is an unseen task in VLRLA's split. This task-specific
initializer cannot establish an unseen-to-the-motor benchmark claim. Before
the six-task campaign, verify a suitable common motor, its training exposure,
normalization and competence across the six source tasks. Record any departure
from the paper rather than implying a matched initializer is available.

## Method boundary

VLRLA-bridge freezes the VLM and action expert, learns a recurrent low-rank
KV residual using flow matching, and retains memory across deployment
episodes with fixed weights. Noise steering remains our primary interface;
it is a different integration method. Both may share the tasks, histories
and evaluation budgets, but the comparison must name the interface used.

Keep previous executed chunks, rewards and both termination flags aligned
causally with the next decision. Preserve history through inner resets and
reset it between trials. RoboTwin source histories randomize the scene within
the same task across episodes; repeating an identical scene is a separate
protocol and must not silently replace the paper's setup.

The paper's main comparison includes GRPO, while its detailed source-history
recipe specifies PPO. Our earlier one-update GRPO smoke verifies a training
signal only. It supplies neither a PPO history corpus nor the VLRLA result.
The reference collector uses eight GPUs; qualify small configurations and
measure throughput before requesting a full collection budget. Keep genuine
successes and failures, source checkpoint identity and chronological update
order. Do not train on the six unseen tasks or select checkpoints with them.

Validate memory/reward controls and matched static/online steering on the
same deployment interactions. A smaller resource configuration is an explicit
reconstruction, pending evidence of comparable behavior. No official VLRLA
implementation or matched multi-task checkpoint has been verified here.

## Recoverability before meta-training

Choose development tasks by measured scope for improvement, not benchmark
difficulty alone. The gate is a competent common motor plus mixed, recoverable
outcomes: steering or an ordinary online learner can improve over native
behavior within the intended target budget. Sample enough paired independent
scenes to distinguish this from random retry variation. Use source-task
development scenes only; lock unseen-task evaluation.

All-success histories carry little outcome-based evidence about recovery.
All-failure histories can be equally unhelpful when steering cannot reach
successful behavior. Do not create arbitrary action corruption just to obtain
zeros and call it an improving source algorithm. Retain genuine attempted
actions and rewards, show the source learner improves on independent probes,
and only then distill complete chronological histories. Alternatively train
recurrent meta-RL on its own trials; it still needs informative outcome
variation and useful actions within the motor's behavioral support.

Mixtures of easy and challenging scenes are allowed, but success-rate bins
and development selection must be declared and reported. Retain the full
predeclared test population, including ceiling and zero-success cases.
