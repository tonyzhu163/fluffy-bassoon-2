# Bounded source-subset target comparison

This development diagnostic uses an explicit one-goal source corpus and its
matched AD/DPT controller exports at25/100/500 optimizer updates. These are
**optimization scales on the same data**, not three source dataset sizes and
not full40 meta-training. DPT500 is compared in the primary robot slice;
DPT25/100 remain optional after all primary groups.

Targets are predeclared LIBERO90 tasks0 and1, initial-state ID0 and scene seed123
for each. There is no semantic or outcome-based admission filter. The primary
methods are native frozen pi0.5, tied Gaussian noise, frozen source pixel-SAC,
AD full/static at25/100/500 updates, DPT full/static at500, and fixed-weight
AD500/DPT500 episode-reset lesions. Each method/case receives an independent0-support
query and one actual support episode followed by a separate reset/query.
The reset lesions isolate memory with the same full-controller weights.
Thus26 primary method/case groups require78 episodes and at most31,980
simulator actions at the400-action
horizon plus10 settling actions per episode.

The single joint reservation is33,046 actions:21,046 from the general reserve
after retaining10 earlier failed-qualification actions, plus12,000 explicitly
escrowed from the490k PPO pool. Future, unlaunched full/reset PPO allocations
are reduced equally from200k to197k each;500 PPO actions remain unassigned.
The benchmark1M total and every active/completed reservation remain unchanged;
target data remain evaluation-only. There are zero motor,
controller or critic optimizer updates at deployment. Query randomness is paired
across methods/support budgets and is independent of the number of random
support actions. Motor and controller parameters are checked unchanged. Only
completed support feedback may enter a query's context. A trained static
controller uses its own matched optimization; the episode-reset lesion reuses
the full-controller weights and clears memory at episode boundaries.

Only after every primary group finishes may the predeclared additional controls
run:DPT full/static at25 then100 updates on task0 then1. A new0/1 group
requires a complete worst-case1,230-action
reservation. Optional groups never displace a primary comparison. Baselines
are collected once per target case and reused as the reference at all scales;
plot reuse is not counted as new interactions.

Frozen source SAC is restored from the same source teacher checkpoint used by
the corpus. It uses the authors' pixel64+proprioceptive8 actor, while the AD
controllers use fixed4104D VLA-prefix features. It is a **source-trained frozen
baseline**, not ordinary target-online DSRL; that adaptation baseline still
needs a separate targeted experiment. The source actor is not presumed expert.

`vla_source40_subset_evaluate` owns at most one original LIBERO90 server process
and switches between tasks. It binds asset hashes, source/motor/export hashes
and the complete requested comparison plan. A write-ahead journal charges
actual and unresolved primitive/settling actions jointly before RPCs. The
controller state and RNG are saved at complete episode boundaries; completed
group receipts and a root progress cursor permit exact-pinned continuation.
Interrupted interactions remain charged, and a group with unmatched interrupted
cost is flagged ineligible for a budget-matched primary claim. No target trace
is admitted to source training.

The scale manifest is hash-bound and comes from the explicit subset trainer.
The executable gate remains separate from production full40 source coverage.
A minimal allocated invocation is:

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_subset_evaluate \
  --scales <subset-experiment>/scales.json --scales-sha256 <exact-file-hash> \
  --output <fresh-evaluation> --checkpoint <official-pi05-libero> \
  --openpi-root <qualified-openpi> --dsrl-root <authors-pinned-dsrl> \
  --libero-root <pinned-original-libero> --simulator-python <sim-python> \
  --simulator-pythonpath <explicit-simulator-paths> \
  --source-commit <exact-clean-commit> --expected-hostname <allocated-host> \
  --max-total-actions 33046 --max-elapsed-seconds 2400
```

Use the same arguments and final `--resume` to continue its original output
and action reservation. Time is bounded per invocation by the allocated
launcher; time pressure may leave a partial matrix, which is reported explicitly.
This tiny development comparison cannot establish full-suite transfer,
high final performance, or a negative conclusion about meta-learning. Report
query success/steps, support costs, case-level results and coverage separately.
