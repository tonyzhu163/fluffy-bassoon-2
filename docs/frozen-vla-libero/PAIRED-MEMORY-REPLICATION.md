# Paired memory replication

The bounded replication fixes LIBERO90 task0, nominal-state IDs1–6 and scene
seed123 before their outcomes are observed. It uses the immutable AD/full
500-update candidate with file SHA256
`ce0b36abf18c33268335816eea34b3d4d329e40a34fa60f1c97da5cf8437e654`.
The hash-bound source-scale manifest and the official frozen OpenPI40 motor
remain fixed. No motor or controller optimizer exists in deployment.

Each case executes one actual support episode and stores its complete context.
It then executes two separately reset query branches with exactly the same
query RNG seed: one restores the actual support context; the other starts an
empty wrapper around the same trained policy. Both retain normal recurrence
within the query. This is a **counterfactual paired probe**, not two independent
deployment trials. The shared support is physically collected and charged
once; both query branches are physically executed and charged.

All support outcomes are retained, including success. Actual nominal-state
hashes, initial public-observation hashes and every initial RGB/wrist/proprio
array must agree across the three resets within a case. Distinct declared
state IDs must produce distinct nominal-state hashes. Neither condition
selects cases by success or failure.

At horizon400 with10 settling actions, six complete triples cost at most
`6×3×410=7380` simulator actions. One joint8000-action ledger covers all known
actions, settling and unresolved RPC intents, including interrupted attempts.
The launcher escrows8000 actions from the existing PPO pool; it does not
increase the benchmark's1M ceiling or alter active trajectories. Future
allocation changes needed to fund that escrow must be recorded before launch.

```sh
bash scripts/diagnostics/vla_source40_memory_replication.sh <exact-SHA> <fresh-run-id>
```

Run inside one allocated GPU job capped at25 minutes. The module has a1200s
invocation ceiling. Qualification runs during the first support reset and adds
no extra physical reset. Full episode traces and per-RPC input/noise/action/
outcome archives are retained. Per-episode context files preserve recurrent
state, controller generator, motor RNG, sampler and global RNG. The complete
episode recovery stores the next case/branch cursor and original support
context. An interrupted query resumes from that boundary without recollecting
an already committed support; all partial physical costs remain charged.
Append `--resume` to the launcher only for the same source, run and ledger.

The result reports paired full/reset successes, shared support outcomes,
committed and interrupted costs, initial-array equality and frozen parameters.
A complete six-case result addresses the use of composite support context on
one goal. It does not establish numeric-reward causality, broad task transfer,+or reliable sample-efficiency/topline superiority over DSRL.
