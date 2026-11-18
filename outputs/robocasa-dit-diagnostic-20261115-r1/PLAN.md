# DIT teacher, advantage and history diagnostic

The user authorized checking whether weak PPO teachers explain the result,
whether learned advantages distinguish successful behavior, and whether history
changes decisions usefully. This is a new offline diagnostic of frozen saved
models and data; the completed 230 GPU-hour campaign remains closed.

- Compute: Baige development Linux CPU, four threads, zero GPU-hours. No fits,
  teacher collection or new simulator episodes. Preserve all source artifacts.
- Models: final three- and six-task DIT and IC-IQL value estimators. Use the
  archived native action-flow intervention measurements for all eight actors.
- Value panel: all 16 held-out streams per task, collection updates 0, 9, 19,
  29, all 12 queries in each selected physical episode. At each endpoint use
  up to 11 preceding complete episodes, within the trained 12-episode window.
  Panel fixed before computing value predictions; no outcome-based selection.
- Compare full history with correctly cleared episode history at identical
  saved observations and executed actions. Report first-query and all-query
  results separately because later observations may already reveal success.
- Report weight distributions, effective sample size, weighted success share,
  and success discrimination within task, update and query position. Bootstrap
  whole held-out streams; repeated queries are not independent samples.
- Read all saved training rewards to describe how much successful behavior and
  improvement the corpus actually contains. Source success rates are descriptive
  and differ from paired SFT/PPO30 evaluation results.
- Outcome association does not identify the causal quality of a particular
  action. Archived flow differences are not new simulator rollouts. These
  diagnostics can identify weak signals, not prove that stronger PPO would fix
  meta-learning or that a particular objective is fundamentally inferior.

Use latest committed main for analysis, recording its pin and hashes. The core
value/history implementation is unchanged from fit commit 4987585e. Validate by
strict checkpoint loading, dataset hashes, finite predictions, the no-prior-
episode full/reset identity, and agreement with the production weight formula.
These checks protect the interpretation of this run; no bookkeeping tests are
added to the repository test suite.
