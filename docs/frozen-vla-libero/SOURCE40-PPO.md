# Fresh PPO source rollouts

The source corpus supports offline AD/DPT training, while direct recurrent PPO
requires fresh interaction from the current noise policy. SAC replay is never
used as an ordinary on-policy PPO buffer. The official frozen OpenPI
`pi05_libero` motor and its prefix-v2 features remain separate from RLinf130.

`vla_source40_ppo` is a **single explicit source task/partition building block**.
It does not yet orchestrate one shared meta-policy across all40 tasks. A series
of newly initialized per-task policies must not be reported as that shared
meta-training experiment. A `--policy` plus matching `--policy-sha256` can load
a prior noise policy with the same motor/features; this transfers policy
weights with a fresh optimizer. It does not transfer a full training process
across tasks. Full same-stream continuation uses `--resume` instead.

## Saved training data

Each raw `trial-*/episode-*` gzip-pickle payload retains:

- Current and next public camera/proprioceptive observations, the current
  fixed4104D prefix features, actual executed32D latent coordinate, full
  repeated H10 initial noise, and the motor actions actually executed.
- Chunk rewards, original primitive rewards, success/done flags, explicit
  success termination versus horizon truncation and executed primitive count.
  Primitive observations between steering decisions are not recorded as extra
  policy observations; the full executed action chunk is retained.
- Actual PPO observation tokens, pre-tanh action samples, old squashed-action
  log probabilities, derived base-Gaussian log probabilities, old value
  estimates, trial/attempt boundaries and recurrent reset masks. Reward from
  decision t is consumed only at decision t+1.
- The exact CPU recurrent cache before the episode (`None` is the zero trial
  state), the behavior-policy update and checkpoint hash, source contract,
  task/instruction/case/reset provenance and chronological trial/episode IDs.

The complete `learner/traces/trial-*.pt` keeps the whole unpadded trial with
zero initial recurrent state, policy RNG before collection, immutable behavior
policy hash and terminal-trial bootstrap value0. Earlier episode endings do
not terminate RL² credit; feedback and memory persist into the next attempt.
The full trial and its policy checkpoint are the PPO input, not independently
shuffled episode files. Returns/advantages are derived from stored rewards,
values, masks and the recorded PPO gamma/GAE configuration; they are not new
environment interactions.

[Paired PPO corpus export](PPO-CORPUS.md) converts committed raw trials into
standard AD/DPT-style datasets with a separately pinned source-trained teacher.
It rebuilds teacher memory from actual history and needs no new simulator/VLA
queries. AD joins same-task trials across real PPO updates; a fixed-policy
trial alone supports DPT/behavior cloning, not a changing learning history.

Before each optimizer update, replaying the sampled pre-tanh actions with the
behavior policy must reproduce the recorded log probabilities within1e-4.
The default implementation performs one full trial per PPO batch and four
epochs, with gamma1 and GAE lambda1. These are this pilot's declared settings,
not a reproduction of VLRLA's much larger rollout batch. `--memory-mode episode`
trains the same architecture with memory reset at episode starts; this control
requires its own fresh interactions and action allocation.

## Physical accounting and recovery

Every reset and action RPC is journaled before execution. The hard
`--max-total-actions` ceiling covers actual primitive actions, all settling,
qualification resets, retries, and unresolved RPCs charged at their full
requested length. Before starting a K-attempt trial the collector reserves
`K*(400+10)` actions. It completes whole trials; a remaining short reservation
is not filled by partial trials. The initial qualification also costs10
settling actions, which are reserved and charged separately.

`learner/latest.pt` atomically contains policy, optimizer, trial sampler,
Python/NumPy/Torch/CUDA RNG and motor PRNG at a complete-trial boundary.
Completed episode files from an interrupted trial are retained. On restart,
an uncommitted trial may be recollected; its earlier physical actions stay
charged and its old files are renamed rather than counted as fresh learning
data. Mid-episode simulator recovery is not claimed. The validation partition
uses nominal states40–49 and an explicit frozen training policy, saves raw
rollouts and RNG recovery, and performs zero optimizer updates. Training uses
states0–39.

Run only inside an allocated Linux job after the actual GPU probe and simulator
qualification. The launcher must expose the correct JAX OpenPI runtime and
qualified Torch CUDA overlay. The source commit/worktree and model receipt are
checked before interaction. A minimal training invocation is:

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_ppo \
  --suite libero_spatial --task-id 0 --partition train \
  --checkpoint <official-checkpoint> --openpi-root <qualified-openpi-root> \
  --output <fresh-output> --source-commit <exact-source> \
  --expected-hostname <allocated-host> --max-total-actions <stream-reservation> \
  --max-elapsed-seconds <this-invocation-ceiling> --updates <requested-updates>
```

The time ceiling applies to an invocation; the original action cap persists
across every continuation. Scheduler time/attempts must also be metered by the
campaign launcher. The 1M benchmark budget is shared by SAC, PPO, validation,
settling and retries. This building block neither authorizes a separate1M per
method nor proves any robotic, memory or sample-efficiency benefit.
