# Can offline joint training teach a locked policy when to use execution history?

The original motion-feedback controller learned to track source-arm displacement and lost four of five previously successful Kinova3 starts. This experiment instead supervises actions from successful source demonstrations. We test whether training an existing part of π₀.₅ alongside an adapter helps the robot act on its own recent observations after the arm changes. All trainable weights are frozen during evaluation. An observation/issued-command history buffer may change at deployment. Its initial state must be reset for each trial.

| Condition | Original π₀.₅ action-head weights trained offline | History adapter trained offline |
| --- | --- | --- |
| Original | No | No |
| Policy only | Yes | No |
| Adapter only | No | Yes |
| Joint | Yes | Yes |

Use a parameter-efficient update to the stock action-expert output projection and a small causal history branch. Train against expert actions using the source action-generation objective and shared examples, ordering, noise, held-out split and budget for all fitted conditions. Prefer the native flow objective wherever feasible. Report actual changed parameter paths and hashes. A zeroed or temporally displaced history condition for the locked joint model tests whether any difference depends on history.

Use independently sourced successful demonstration actions. Reserve simulator arms used for native evaluation from fitting and checkpoint selection. Register concrete source tasks, demonstrations and target starts before training. Keep new scores independent of previously reported layouts. Report native reward on paired unshifted arm transfers and results on source-arm retention. Compare joint against each single-component control and the original. Record rescues and new failures alongside denominators and uncertainty. A fitted model with worse retention, inferior joint success, or equal scores after suppressing history would count against the corresponding claims. Source imitation and gradient checks alone establish neither adaptation nor native transfer.

Run on Baige Distributed Training. The initial allocation envelope is 12 GPU-hours across collection, fitting and native evaluation, with CPU preparation in parallel. Record receipts per stage. No use of test-arm fit examples, target validation, optimizer, gradient steps, replay updates or checkpoint selection at test. Fixed normalization and read-only checkpoint digests must agree before and after evaluation.

## What the co-trained policy computes

The stock motor produces a hidden feature for each of ten proposed actions. Its existing final linear layer converts each feature into a denoising velocity. The history branch takes four completed transitions. Each transition contains the issued command and measured tool and jaw motion. A 16-unit layer converts that history into ten offsets, one per proposed action. The joint variant applies a learned rank-four update to the stock output matrix and adds the history offsets at each denoising step. Its two trainable components contain 10,176 scalars in total. At deployment a new observation changes those offsets through an ordinary forward pass. The trained matrices stay fixed throughout every rollout. [Exact layers and dimensions](model/README.md) · [Source demonstration and history fields](data/API.md)

The original three evaluation tasks are held out of this additional fitting. A separate eight-case paired Panda/Kinova3 panel uses two tasks present in the fitting source to distinguish body transfer from transfer to other tasks. Both panels use the same locked checkpoints. [Held-out-task panel](evaluation/PANEL.md) · [Task-matched panel](matched-task/PLAN.md)
