# Evaluation status

The records below were made in fluffy-bassoon before the split
(see [PROVENANCE.md](../../../../PROVENANCE.md)).

No paper benchmark reproduction or robot-policy result is qualified yet.
Equation-oracle, recurrence, mask, gradient and state-restoration tests are
defined in `tests/test_continuous_memory_machine.py`.

At source `4fdefab73ddff600c6a18ba495402023838fd2b5`, the school test coordinator
reported all **57 tests plus 27 subtests passing** in secondary-school
job **47024** on 25 October 2026. The aggregate covers
`tests/test_continuous_memory_machine.py`, `tests/test_ctm_rl.py` and
`tests/test_gpu_identity_mapping.py`; the job completed with exit **0:0**.
These are reported server results, not tests rerun by this Mac documentation
session.

The earlier non-leaf `deepcopy` fixture failure is resolved by detached tensor
clones. CMM equation-oracle and state/gradient coverage passed in the final
suite. The balanced native PPO run in the same job used an existing CTM policy,
not this separate CMM core. It does not qualify a CMM robot policy or reproduce
the paper's benchmark results.
See the [CTM RL evaluation](https://github.com/tonyzhu163/fluffy-bassoon/blob/0df7d026cb72d1d047bbb261ef95dc1f5d2da9fa/src/fluffy_bassoon/rl/EVALUATION.md) for those
separate numerical/native engineering results and their limits.

Evidence: secondary-school
`/cluster/nvme5/zhujiayang/fluffy-bassoon/outputs/ctm-ppo-balanced-20261025-r3`.
