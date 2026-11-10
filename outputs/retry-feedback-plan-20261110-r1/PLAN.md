# A general in-context learner for π0.5 on RoboCasa atomic tasks: plan

Status on 2026-11-10: proposal. Nothing is launched and this document approves no compute. It builds on the controls in the [Nov 9 feedback plan](../vlrla-feedback-plan-20261109-r1/PLAN.md) and replaces the earlier draft in this folder.

## Question

After meta-training on many tasks, does a history-conditioned learner improve π0.5's next attempt on a held-out task with every weight fixed? Does feedback from a reviewer add to that?

## Design

1. **Meta-training.** A decision-pretrained transformer reads the history of earlier attempts (observations, executed actions, outcomes) and steers π0.5 through the KV or velocity interface. π0.5 receives a low-rank update on its action expert in the same run, so it responds to the steering.
2. **Freeze.** π0.5, the low-rank update and the transformer are all fixed.
3. **Deployment.** On a held-out task, each completed attempt is appended to the history. No weight changes.

No condition is injected into the tasks. What history can reveal is the task and scene themselves, so generality depends on the number and spread of source tasks.

**Changes from the recorded design**

- π0.5 is lightly adapted, not frozen. [PAPER.md](../../docs/PAPER.md) and [METHODS.md](../../docs/METHODS.md) still say frozen and are not edited here; this needs a decision. A frozen-only fit stays as an ablation.
- The source corpus covers about 50 atomic tasks with short histories, in place of five tasks at about 10M controls each.

## Proof of concept first (target: under 50 GPU-hours in total)

Nothing in the 50-task corpus starts until both gates pass.

**Gate 1: is there anything in history to learn? (about 15 GPU-hours, inference only)**
- Run contract and scripts: [retry-history-gate1-20261110-r1](../retry-history-gate1-20261110-r1/PLAN.md).
- Collect 8 tasks × 16 scenes × 8 base-policy attempts from matched starts, 1,024 episodes. The matched-start harness measured 1.9 GPU-hours per 128 episodes, so 32 scenes would have cost about 30.
- Per task, report single-attempt success, best-of-8 success and scene repeatability.
- On CPU, fit a small predictor of whether an attempt succeeds, from the scene and its first action chunk, with and without the earlier attempts and their outcomes. Hold out whole scenes.
- **Pass:** adding earlier attempts improves held-out prediction on at least some tasks. If it does not, earlier attempts carry no usable signal and the learner has nothing to read.

**Gate 2: does a small learner use history? (about 30 GPU-hours)**
- Meta-train on 6 of the 8 tasks using the gate 1 episodes plus best-of-several labels, with the low-rank update.
- Test on held-out scenes of the training tasks first: own history, donor history, reset and native retries, one support, about 200 paired cases.
- **Pass:** own history beats donor history with a paired 95% lower bound above zero.
- The two held-out tasks are reported as a first look at transfer, not as a gate; six training tasks are too few to expect generality.
- **Fail:** if history is not used even on training tasks, more tasks will not fix it. Stop and revisit the label construction or the interface.

Training cost is unmeasured; time one short fit before committing gate 2.

## How meta-training examples are built

- **A trial** is one task and one starting scene with several attempts.
- **Context:** the earlier attempts in that trial, with observations, executed actions and outcomes.
- **Query:** a later attempt in the same trial that succeeded.
- **Label:** the actions executed in that successful attempt, at its own observations. No action is assigned to states the labelled policy never visited.
- **Loss:** π0.5's flow-matching loss on the label, through the action expert, into the low-rank update and the transformer.
- **Why it is meta-learning:** batches mix many tasks and scenes, and the same weights must do well on all of them. The only input that differs between a trial where approach A works and one where approach B works is the history.
- A qualified RL teacher, where one exists, adds labels at the learner's own query states, which is the stricter DPT form.

## Source corpus: 50 atomic tasks (only after both gates)

Naive cost for reference: about 25 GPU-hours per separate teacher at 3M controls (from the Counter and Faucet runs), so about 1,300 GPU-hours for 50, with three of five past teachers qualifying.

**Step 1: screen every task (inference only).**
- A few hundred base-policy episodes per task on fixed scenes, with repeats from matched starts.
- Record success rate and how repeatable failures are at a fixed scene.
- Drop tasks near 0% or near 100%. Estimated cost: about 50 GPU-hours.

**Step 2: best-of-several labels on every surviving task (no RL).**
- Run each scene several times. Where at least one attempt succeeds, a successful attempt from the same start is the teacher label for the failed ones.
- About 2,000 episodes per task; roughly 230 GPU-hours for 50 tasks at the measured 2.3 GPU-hours per 1,000 episodes.
- Limit: never better than the best of several base attempts, and scenes that always fail get no label.

**Step 3: one multi-task RL teacher for persistent failures.**
- A single language-conditioned run across surviving tasks, training only a low-rank update or small steering policy on the frozen base.
- Short budget per task, then continue only tasks that improve.
- Qualify each task's teacher against the base on fresh paired cases before using its labels.
- Prerequisites: repair the critic rollout and replay mismatch, and confirm RLinf supports multi-task PPO (unchecked).

**Splits and transfer tests.** The learner trains on all screened atomic tasks in the `pretrain` split. Transfer is tested on:

1. **Unseen kitchens and objects:** the same atomic tasks in the `target` split. Primary test; covers every task.
2. **Unseen composite tasks:** Composite-Unseen, which neither the learner nor the Human300 π0.5 was trained on. Headline transfer; base success may be near zero, and history cannot supply a missing skill.

Keep all test histories, outcomes and labels out of fitting and selection. Human300 π0.5 was pretrained on the atomic tasks, so only the composite tier is new to the motor as well as the learner. Holding out a few atomic tasks remains an option if composites prove too hard to show anything.

## Code to write

| Item | Where |
| --- | --- |
| Low-rank update on the action expert, trained jointly | Flow bridge trainer; none exists in `src` or `benchmarks` today |
| Parity check: adapted model with empty history matches native π0.5 on a fixed bank | New evaluation |
| Optional feedback field in the history row, with a "missing" encoding seen in training | [vla_history.py](../../src/fluffy_bassoon/meta_rl/vla_history.py) |
| Best-of-several corpus builder | Next to the existing DPT corpus construction |

## Feedback (optional arm)

- **Default:** native success or failure only.
- **Reviewer arm:** add a grade or diagnosis per attempt. In simulation, write diagnoses from simulator state so every training attempt has one; score a video reviewer against them. Flora stays reserved for physical-robot testing per [FLORA-PPO.md](../../docs/FLORA-PPO.md).
- Feedback present at deployment must be present in the same form in meta-training.

## Evaluation

Both transfer tiers, native success, paired by case. Controls from the Nov 9 plan, unchanged:

| Condition | Tests |
| --- | --- |
| Own history | The method |
| Donor history, matched on outcome and length | Whether the particular experience matters |
| Episode reset | Reliance on any retained input |
| Reward masked | Role of the outcome signal |
| Separately trained memoryless fit | Gain from offline training alone |
| Native retries | Gain from another try |
| Empty feedback slot, shuffled feedback | Reviewer arm only |

- **Primary endpoint:** own minus donor history after one support, at least 5 points with a paired 95% lower bound above zero, across three fits and 200 fresh cases per task.
- Report success at 0, 1, 2, 4 and 8 supports, first-attempt and cumulative separately, and against delivered actions.
- **Reference:** online RL on the held-out task at equal deployment budget.

## Risks

- **History ignored again.** Earlier full-versus-reset comparisons were flat, and VLRLA with six source tasks matched online RL only. More tasks is the intended fix but is not guaranteed.
- **No diagnosis if it fails.** Without a controlled case, a null result cannot separate "method failed" from "tasks offered nothing to learn". A small controlled check outside the method would resolve that; it is left out by decision.
- **Weak labels.** Best-of-several labels cap at base performance on persistent failures; the RL teacher covers those only where it qualifies.
- **Base degradation** from the low-rank update, guarded by the parity check.

## Budget

Rough total 400 to 600 GPU-hours for the corpus, before meta-training and evaluation. All of it exceeds the 50 GPU-hour limit and needs approval. Measure rollout throughput in step 1 and re-price before step 2.

## Open decisions

1. Approve dropping "frozen" and update PAPER.md and METHODS.md.
2. Approve the 50-task corpus and its budget, superseding the five-task 10M-control campaign.
3. Whether to include the reviewer arm in the first paper.
4. Whether to keep one controlled diagnostic outside the method.
