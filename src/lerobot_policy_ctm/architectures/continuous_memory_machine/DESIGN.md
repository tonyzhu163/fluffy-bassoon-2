# Continuous Memory Machine recurrent core

Source: Regan et al., [Continuous Memory Machines, arXiv:2610.07907v1](https://arxiv.org/html/2610.07907v1), Section 3.2 and Appendix A.
This is an independent implementation of the published equations. The linked
authors' repository (`SakanaAI/continuous-memory-machines`) still returned 404 on
2026-10-28; code parity is unverified.

`core.py` is the repository's only CMM. `fluffy_bassoon.meta_rl.cmm.CMMCore` is a
thin wrapper with the compact trainer's keyword API: it steps one observation
and returns synchronization features instead of a readout.

| Paper contract | Implementation |
| --- | --- |
| Eq. 1, synapse | Retained CTM SynapseUNET, default depth 8, bottleneck min(16, d); optional explicit replacement |
| Eqs. 4–5, memory update | Bidirectional attention over sink, LTM, STM; shared slot positions |
| Eqs. 7–12, block | Pre-norm; positions in Q/K only; zero sink V; bias-free projections; width-d SiLU FFN; dropout on the attended values and after W_O |
| Residual gate | Fixed one, or one learned zero-initialized scalar per block shared by both residuals |
| Eq. 13, recurrence | Layer-normalized transformed LTM persists |
| Eq. 14, NLM | Private bias-free history-to-two-output weights per neuron, no temperature, GLU, then LayerNorm |
| STM lifetime | Raw rolling pre-activations persist; transformed STM feeds NLMs and is discarded |
| Initial state | Zero activated/STM buffers; learned LTM N(0, .01²); zero learned sink; positions N(0, .02²) |
| Appendix A.3 | d fixed unique ordered neuron pairs from a seeded generator; linear weight-sum denominator; +1e-8; learned RMSNorm |

State keys are `activated`, `trace` (raw STM), `long_term` (LTM), `alpha` and
`beta` (synchronization numerator and denominator). `step(observation, state)`
accepts `[B,input_dim]`; `forward` scans `[B,T,input_dim]` with per-lane resets
and padding. Incoming state is never mutated or detached. `ticks` repeats the
step on one observation and accumulates synchronization per tick.

Explicit implementation choices where the paper is silent: per-pair decay
r = clamp(decay_params, 0, 15) initialized at 0, the CTM convention (uniform
averaging at the start of training); one tick per input unless configured;
normalization epsilons and dropout are saved configuration fields. Eq. 3 uses
(t+1-τ) and eq. 15 (t-τ) in the decay exponent; under the linear normalization
the common factor cancels, so the two agree here.

`memory_layers=0` (requires zero LTM) is the CTM-style control used by the
compact screens: neurons read the raw trace, with no sink, positions or blocks.
It is not the paper's CTM baseline. `ltm_slots=0` with one memory layer is the
paper's CMM-STM ablation.

Consolidation on 2026-10-28 changed the compact core: its NLM no longer has
Sakana SuperLinear's bias and temperature, its pairs are seeded, and attention
dropout moved from the attention probabilities to the paper's positions. Compact
CMM/CTM-control checkpoints saved before this change do not load at later
sources; evaluate them at their recorded source commit. The full core's decay
initialization changed from softplus r=1 to r=0.

This vector-input core has its own parameter/state layout. Existing CTM/VLA
checkpoints cannot be reinterpreted as CMM weights. A robot encoder/action
adapter and its initializer require a distinct saved policy identity and
qualification; the current CTM PPO entrypoint intentionally does not accept CMM.

## Calibration

`fluffy_bassoon.meta_rl.fewshot_regression` reproduces the paper's few-shot
linear regression (§4.3, App. D.5, Table 2) for three variants with published
rows: CMM, CMM-STM and CTM-Ext. Its docstring lists the paper's stated settings
and the choices made where the paper is silent. Interpret CMM comparisons
elsewhere only after this calibration lands within the published spread.

Launch one process per run:

```bash
uv run python -m fluffy_bassoon.meta_rl.fewshot_regression --source-commit COMMIT \
  --output-dir OUT/fewshot-cmm-k20-s0 --variant cmm --shots 20 --seed 0 [--device cuda]
```

Variants `cmm`, `cmm_stm`, `ctm_ext` × seeds 0, 1, 2 at K=20 is nine runs.
Paper targets (query accuracy %): CMM 30.63 ± 0.24, CMM-STM 13.03 ± 0.61,
CTM-Ext 8.00 ± 0.07. Full length is 1,000 epochs with exactly 10,000 examples
per epoch (157 updates at batch 64, final batch 16). Measure update time with
`--epochs 1` and report the projected cost before launching the set. CMM tests:
`tests/test_continuous_memory_machine.py tests/test_meta_rl_cmm.py
tests/test_meta_rl_fewshot.py tests/test_meta_rl_copy.py tests/test_meta_rl_ppo.py
tests/test_meta_rl_train.py` (and `tests/test_meta_rl_cuda.py` inside a GPU step).

Parameter counts: at the few-shot layout (M_S 10, M_L 32, J 1, synapse depth 8,
input projection to d, scalar head) this core gives 498,449 parameters at d=190
against the paper's 496,129, so the depth-8 U-Net synapse is consistent with the
paper. The earlier Copy mismatch (about 690k at d=224 versus 486,474) therefore
more likely reflects the width assumed for Copy than the synapse.
