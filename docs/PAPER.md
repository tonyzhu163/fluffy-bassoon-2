# Research question and evaluation

Does a history-conditioned policy improve its next attempt on a held-out task while every model weight stays fixed? The key comparison is the **same frozen learner with its own prior experience retained versus cleared**, at the same query attempt and on matched cases. To test whether the *particular* experience matters, also compare own history against a matched donor history: reward and task can be the same while the earlier interaction differs. The [simulation experiment plan](../outputs/vlrla-feedback-plan-20261109-r1/PLAN.md) specifies the primary own-versus-donor endpoint, reference controls and fixed support budgets. High first-attempt performance measures initial competence, not adaptation.

## Training and deployment

Use RoboCasa365 atomic tasks and the RoboTwin source split for offline history learning. Keep complete source streams together when separating training and validation. Keep held-out evaluation histories, outcomes and teacher labels out of fitting, normalization, early stopping and model selection. Record the base motor’s pretraining/SFT exposure separately: a task held out from the learner may have been seen by the motor.

Use native simulator rewards for source learning and completed feedback in the student history. Record success termination, truncation, visual cues and action duration when interpreting reward-blind controls. RLinf supplies ordinary online RL baselines and can produce chronological teacher data. Those teacher policies may change during collection. During adapter training the original π0.5 vision-language model and action expert remain frozen; gradients may pass through the expert into a trainable adapter. During deployment the history model and adapter also stay frozen. See [Methods](METHODS.md) for objective-specific target and interface contracts.

A trial contains multiple real attempts at the same task. Reset the simulator between episodes and reset learner memory between trials. At a query, the learner may see the current observation, instruction and previous executed actions, rewards and boundaries. The query's action, outcome and future observations cannot enter its own context. Keep model-space commands separate from what the simulator actually executed.

## Evidence required

- Compare native SFT, an ordinary RL teacher and the offline learner on matched cases. For the learner compare retained history to episode-reset, static and reward-blind controls where applicable. Keep motor, task, seed, horizon and physical-control budgets comparable.
- Report native success at each query attempt. Show first-attempt performance and cumulative success separately; more chances alone can raise cumulative success. Pair uncertainty by case and report the number of independent fits.
- Report simulator controls, executed action chunks and source/fitting/evaluation cost. Measure success independently of the training reward. A high video-reward score, a lower offline loss or a successful source PPO run cannot establish held-out adaptation.

RoboCasa composites are a later transfer evaluation, not part of current source fitting. Physical-robot trials would require [Flora qualification](FLORA-PPO.md) against independently judged outcomes and separate reporting of physical actions, attempts and feedback cost. The longer-term motivation is sample efficiency on real robots; neither simulator results nor a proposed real-robot experiment establish that claim.
