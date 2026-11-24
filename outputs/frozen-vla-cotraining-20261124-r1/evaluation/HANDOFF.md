# Native co-training evaluation on Baige

The evaluation compares the four offline training conditions and two interventions on the identical joint checkpoint. The 108-episode manifest uses the source-held-out Goal and LIBERO-10 suites. Source train and held-out demonstrations come from Object and Spatial. All deployment inputs are public observations, actually issued commands and paired diffusion noise. [Fixed panel](PANEL.md) · [Source data and split](../data/RESULT.md) · [Model and checkpoint contract](../model/API.md)

## Deployment

Root owns GPU submission. Use Baige **分布式训练** with one RTX 4090 or RTX PRO 6000D, 12 CPUs, 64 GiB memory, the existing shared PFS mounted at `/mowenjie`, and a scheduler wall limit of four hours. The launcher inherits the allocated device, verifies the model and count, maps the renderer to its EGL device and bounds runtime at 14,340 seconds plus 10 seconds kill grace. The historical 24-episode native feedback job used 0.203 GPU-hours on one 4090. Timing with the co-trained sampler is unmeasured. [Native timing](../../frozen-vla-tracking-feedback-20261123-r1/RESULT.md)

Use the exact detached checkout and command in the handoff file. The general form below uses the successful training attempt `gpu01` and a fresh evaluation attempt of the same name in its separate directory.

```bash
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1
tree="$run/evaluation/code-$PIN"
bash "$tree/outputs/frozen-vla-cotraining-20261124-r1/evaluation/deploy.sh" "$PIN" gpu01 gpu01
```

The operator may supply the successful training recovery ID as argument two and a fresh evaluation attempt ID as argument three. The command checks the trainer exit receipt and reads the fit digest it printed on successful offline completion. Admission compares that digest, the staged data manifest, input conversion and history source, live injection code, parameter groups and actual selected suites. Inference reads the stock policy and fitted arrays. `frozen-before.json` and `frozen-after.json` hash the resident NNX parameters, fitted head arrays, normalization, fit and head files. Both files must agree before the evaluator releases scores.

The evaluator reuses the pinned native simulation and OpenPI motor environment at `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-transfer-20261123-r1/src-a8a997e7d06bb131d3f9710204c58e106f0f0904`. It calls the qualified stock observation converter with its original quaternion convention and clips each executed seven-dimensional proposal to the native action bounds. It reconstructs each history entry from the measured position and finger joints on either side of the clipped command. A fresh buffer starts after the released reset and settling. The lagged intervention gives the joint model the history vector seen one replan earlier.

## Readback

The attempt directory contains the GPU probe, trainer identity, launch and exit receipts, immutable parameter digests, and one JSON tape per episode. The tapes retain exogenous bank and observation hashes, all issued controls, response coordinates, actual native reward at every control and terminal flags. Failed rollouts must reach the released 300-control Goal or 520-control LIBERO-10 horizon. Only a complete panel with matching before/after hashes is scored. An unmatched reset may retry up to twice before any inference. Each accepted attempt is logged. No policy outcome is used to retry a case.

`native-summary.json` gives the four training-cell scores, paired rescues/losses for joint against each comparator, the additive-scale interaction and the aligned/zero/lagged comparisons. Rows keep task, released start, return, controls and success. `arms` reports paired 95% bootstrap intervals resampling starts within each fixed task. Six Panda starts and 12 Kinova3 starts share one fit and one bank per case. Sparse panels can give a zero-width empirical interval. `matched_arms` presents within-arm treatment effects for the six common task/start pairs.

`first_changes` locates the first differing proposal and first issued control. Each joint episode also records unissued counterfactual proposals at control 10, using aligned, all-zero and five-control-lagged history on the same current observation and noise. This measures dependence on the branch input. The terminal native return comparisons measure performance. A history intervention changes command history and jaw-width information along with measured motion, so its effect alone cannot identify a response to arm dynamics.
