# Fixed100 readout: the robotic-performance objective remains unmet

The complete joined outcome grid contains486method/case groups and522episodes across all90LIBERO90 goals, with zero target optimizer updates. Fresh native succeeds28/90(31.1%); ADfull/static29/90each(32.2%); DPTfull/static28/90each(31.1%). This is a small mixed paired change, not demonstrated high topline performance or high sample efficiency. Interpretation uses the saved complete joined results; the separate owner’s final raw-byte/NumPy integrity audit remains in progress at this analysis snapshot and is not a new scientific gate.

## Paired outcomes matter more than the aggregate tie

|K0query|Successes|Student succeeds/native fails:taskIDs|Native succeeds/student fails:taskIDs|
|---|---:|---|---|
|AD full|29/90|33,59,82|77,79|
|AD static|29/90|29,33,53,59|64,70,77|
|DPT full|28/90|33,59|64,77|
|DPT static|28/90|33|77|

ADfull's+1success is three gains and two regressions. ADstatic's+1is four gains and three regressions. Equal full/static totals do not mean identical behavior: ADfull wins on64/70/82,static on29/53/79; DPTfull wins on59,static on64. These are descriptive paired outcomes on the entire predeclared bank. None of these taskIDs becomes a selected training/evaluation set or a checkpoint-selection rule.

The static control uses the same Transformer architecture, source data and optimizer budget while removing history input/state. Therefore these results supply no net history advantage atK0 and do not identify an architecture advantage over an MLP/SAC. K0full can still use within-episode history. A reactive Transformer can learn useful steering; an equal static score does not itself prove implicit meta-learning. Ordinary target-onlineDSRL and recurrentPPO are outside this wave, so neither an architecture comparison nor superiority over those baselines is measured here.

## Actual one-support/query panel

The fixed nine tasks are0,10,…80. EachK1group executes one complete support followed by a separate fresh-reset query with fixed parameters. Query seeds are paired by the existing protocol; support costs are separate.

- ADfullK1and same-weight reset each succeed6/9, with **identical successful IDs10,20,30,50,60,70**. ADfullK0and nativeK0 already have that exact6/9success set. All three failed ADsupports(0/40/80) are followed by failed queries for both full/reset.
- DPTfullK1scores5/9versus reset6/9. The only discordant task is60,favoring reset; its support succeeded. DPTfullK0/nativeK0already score6/9. On70,a failed support is followed by success for both full/reset, so that sequence supplies no positive memory discordance.

Equal success outcomes do not establish identical trajectories or prove that the model ignores context. They do establish **no realized positive cross-episode-history success difference** on this panel. The one DPT discordance cannot establish general harmful memory either. Learned reward use, feedback causality and robust adaptation remain unshown.

ADfullK1spends2006support actions plus2008query actions=**4014actions for6successful queries**. NativeK0uses1961actions for the same6/9success set. This panel shows no interaction-efficiency gain from paying for support. Across the full experiment,36supports cost8602actions and486queries163443,total172045. Conservative worker charge7126GPU-seconds is distinct from actual scheduler duration6846seconds and from source collection/fitting costs; evaluation cost is not source dataset size. All source histories are shared inputs to the four fits, counted once rather than multiplied by four.

## Historical500versus current100 does not isolate optimization or data scale

|Wave|Training corpus|Fixed fit budget|NativeK0|ADfull/static|DPTfull/static|
|---|---|---:|---:|---:|---:|
|Historical broad screen|527K4/2108episodes|500updates|31/90|26/26|26/27|
|Current interim readout|583K4/2332episodes|100updates|28/90|29/29|28/28|

Both the source bank and update budget changed, and the native control changes by three successes between waves. Prior matched observations/noise have already shown cross-process motor decoding sensitivity. The present counts therefore cannot establish that early stopping helped,500updates overfit, more source data improved performance, or scaling failed. Different raw NLL targets and independently trained validation teachers further prevent interpreting NLL as robot competence. Every fixed endpoint remains reported; no outcome-based checkpoint choice is made.

Current source training contains33729decisions over all40goals. The prior source audit found94failures/2108episodes and only10K4cases with a learned-behavior failure followed by learned-behavior success. The added224bridge/qualification episodes all succeeded, so **no additional such recovery pattern is established**. This makes limited corrective/reward-dependent experience a plausible bottleneck. It is not a measured cause: model fitting, history objectives, representation/noise calibration or optimization can also produce similar full/static behavior. DPT's reactive final-actor targets may emphasize current-observation steering rather than an improvement process; AD's actual policy-change histories need not contain useful improvement just because optimizer clocks advance. The present data do not distinguish these explanations.

## One discriminating next contrast and decision

Freeze the **current583K4bank**, source interfaces, architecture, seed and window sampler. Compare fixed100and500exports from **one continuous source-only optimization trajectory**, with matched full/static controls and the same predeclared full90K0/fixed9K1full/reset panel. Include a fresh contemporaneous native control in the same runtime protocol. Keep actual support/query costs and every predeclared case; do not choose cases or checkpoints from these outcomes. This is a proposed scientific contrast, not a new run, implementation or resource request.

That contrast holds data fixed while varying optimizer exposure. Improved full/staticK0with noK1resetgap would support better reactive steering rather than context adaptation. A repeatable fullK1advantage over the same weights reset would be the relevant history signal. If neither improves, automatic expansion of this same collection recipe is not supported by the current evidence; the next design should isolate a specific source-history/objective/noise explanation instead of assuming more data is the remedy. One fit seed and one initialization per goal still limit any conclusion; a positive contrast would need replication before a paper-level claim.

The protocol remains official frozenOpenPI40-source training and the **complete90-task target suite**, with no semantic filtering and no assertion that every90goal is semantically novel. Plus/Pro and full0/2/4/8support readouts remain incomplete. The user’s objective is high robotic sample efficiency and topline performance; this readout has not achieved it.

## Evidence boundary

`paired-counts.json` independently recomputes the pairs above from the owner’s SHA-bound `joined-metrics.json` and `case-results.csv`, retaining all90outcomes. It performs no NN/model/test/simulator work and does not duplicate the owner’s raw trace audit. Producer100fitpin2bc15da; matrixdc9ebd1274170e021906c1f66623a98713405e227229bc022e03eb4c1c92a623; unchanged evaluatore764afdd04c9ea4dcbb7ac3e3e25d245c186dd42; motor digestc565dd3f… . Historical scores/corpus context are from the canonical priorEVALUATION/source-failure receipts and remain separate waves.
