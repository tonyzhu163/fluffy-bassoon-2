# Quick test: RLinf PPO on LIBERO90 task 44, simulator reward vs Flora reward

Draft, 2026-11-01, against main `c729a98`. **Plan only:** nothing has run.
A separate frozen-pi0.5 DSRL run on task 44 (Baige `job-2qn8dpw3yvlb`) is
active. This test shares no outputs, worktree or ledger with it.
This is a single-seed pilot to see whether the result is worth pursuing. It is
not a definitive comparison.

## Question

On LIBERO90 task 44 (`KITCHEN_SCENE9_turn_on_the_stove`, "turn on the
stove"), does PPO trained only on
[`nvidia/Flora-Rewarder`](https://huggingface.co/nvidia/Flora-Rewarder)
outcome labels improve true success like PPO trained on the simulator's success
reward? The simulator check scores every evaluation. Flora is used only as the
training reward in arm F.

## Setup

- **Policy:** [`RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT@6222623f`](../../docs/frozen-vla-rlinf-libero.md).
- **Recipe:** RLinf's pi0.5 LIBERO PPO config, restricted to task 44, with
  the published PPO hyperparameters. To get more updates within a small
  budget, reduce the env count to 32 envs × 4 rollout epochs. That is about
  51k actions per iteration at a 400-step horizon.
- **Episodes in both arms:** success termination is disabled in training, so
  every episode runs the full horizon. Otherwise episode length would leak
  the simulator's success signal into arm F. Each episode gets one terminal
  reward on its last chunk: 1 if the stove is on in the final state, 0
  otherwise.

| Arm | Who decides the terminal reward |
| --- | --- |
| **S** | The simulator success check |
| **F** | Flora's outcome score ≥ τ, where τ is fixed in step 2. The trainer never reads the simulator label. |

In arm F, the simulator label is still logged to a side file for later
analysis, and the trainer never reads that file.

## Steps

1. **Flora smoke test.** Download the pinned revision to Baige NVMe and run
   one task-44 episode through it: up to 64 agentview frames, upright, with
   the instruction. Check that the outcome score parses.
2. **Baseline check, about 64 episodes, no training.** Roll out the SFT
   policy at the full horizon and label every episode with both the
   simulator and Flora. This gives:
   - the starting success rate
   - Flora's agreement with the simulator, and its AUROC
   - τ: the score cutoff that best matches the simulator labels

   **Stop** if the starting rate is under 5% or over 70%, since PPO then has
   little to work with. **Also stop** if Flora barely tracks the simulator
   (AUROC < 0.7), meaning it can't see the stove state. Either way, report
   what was found.
3. **Wire up and test on CPU.** Add a relabel hook after rollout and before
   GAE, using Flora served on its own GPU. Add the full-horizon training env
   setting. Write tests that:
   - arm S, with a stub rewarder returning simulator labels, reproduces the
     plain config
   - arm F's trainer cannot read the simulator label
4. **Run.** One seed each for S and F, from the same commit, seed and
   training states, for about 10 iterations (about 0.5M actions per arm).
   After every iteration, evaluate on 50 held-out initial states using the
   simulator check.
5. **Report.** Write `RESULT.md` with:
   - success curves for S and F on the evaluation states, with Wilson
     intervals
   - Flora-vs-simulator agreement during training in arm F. If "Flora yes,
     simulator no" rises, the policy is learning to fool the rewarder.
   - actions and GPU time per arm, with Flora's compute charged to F

## Budget

- About 0.5M actions per arm, under the 1M-per-learner ledger.
- Check arm S at iteration 10. If S has not clearly risen above its starting
  rate, extend both arms by the same amount, up to 20 iterations (about 1M
  actions each). Do not judge arm F against an S that hasn't learned yet.
- At most 10 GPU-hours in total, including Flora serving. This is within the
  standing limit.
- If step 4's real iteration time exceeds the limit, cut iterations rather
  than the batch.

## Reading the result

With one seed, only a large effect is readable. Possible outcomes:

- F tracks S, and both clearly rise above the starting rate: worth a
  multi-seed follow-up.
- F stays flat while S rises: Flora's labels don't carry enough signal for
  this task.
- F's "Flora yes, simulator no" rate climbs: PPO is exploiting the rewarder.
- S itself doesn't improve: the test says nothing about Flora.
