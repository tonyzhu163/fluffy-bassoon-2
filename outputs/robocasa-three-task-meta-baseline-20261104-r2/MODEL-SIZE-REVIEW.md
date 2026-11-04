# Transformer capacity review

2026-11-04 Asia/Shanghai. User decision: **at least width 256, six layers**.
The launcher now uses **256 width / 6 layers / 8 heads** for all four actors
and the DIT/IC-IQL critics. No scaled GPU job has been submitted.

| Model / paper | Transformer architecture | Source |
| --- | --- | --- |
| Previous local baseline | Width 128, 2 layers, 4 heads; 817,456 actor parameters | Linux CPU module count |
| Current user-selected baseline | Width 256, 6 layers, 8 heads; 5,579,184 actor parameters | [Exact counts](MODEL-SIZES.json) |
| Original AD | 4 layers, embedding 64, FFN 2048, 4 heads; separate modality tokens | [Appendix L, Table 1](https://arxiv.org/html/2210.14215#A12) |
| Original DPT | Width 32, 4 layers, 4 heads in Dark Room and bandit experiments | [Appendix A.2](https://arxiv.org/html/2306.14892#A1.SS2) |
| DIT | Width 256, 6 layers, 1 head; Q/V transformers mirror the policy architecture | [Appendix G](https://arxiv.org/html/2601.20116#A7) |
| IC-IQL / YesQ study | Default width 512, 4 layers; 8 layers for XLand-MiniGrid. AD backbone size reported as approximately 12.5M / 25M | [Appendix F and Tables 32–33](https://arxiv.org/pdf/2502.17666#page=23) |
| VLRLA-bridge | Width 512, 6 layers, 8 heads, FFN 1024, residual adapter rank 64 | [Appendix A.1, Table S1](https://arxiv.org/html/2610.19517#A1.SS1) |
| Vintix | Width 1024, 24 layers, 16 heads, FFN 4096; reported 300M parameters | [Section 2.3.1](https://arxiv.org/html/2501.19400#S2.SS3.SSS1) |

The old 128x2 controller was a small exploratory configuration. It is smaller
than the DIT, IC-IQL and VLRLA references; tiny DPT experiments show that no
universal minimum model size follows from ICRL alone. The selected 256x6 model
has about 6.83 times the old actor parameters and matches DIT's width/depth.
Our IC-IQL implementation uses separate actor/Q/V transformers, with 16,713,650
trainable parameters at 256x6; that sum is not the deployment actor's size.
Frozen VLM/teacher parameters are excluded from these adaptive-model counts.

Capacity adequacy remains an empirical question. The YesQ size ablation finds
moderate widths 256–512 competitive, while excessive depth/width can hurt in its
settings. Before the user's minimum-depth decision, 512x4 was the suggested
capacity check; it is not the adopted configuration. Train/validation curves
and frozen closed-loop evaluation are needed to establish underfitting or an
adaptation advantage. No additional capacity-sweep allocation is authorized here.

Contexts also use different token units across papers. Our controller reads one
H50 decision chunk per token, with a 192-token cache and training windows capped
at 12 complete H600 episodes (144 chunks); raw token counts do not directly
compare to per-control or per-modality contexts.
