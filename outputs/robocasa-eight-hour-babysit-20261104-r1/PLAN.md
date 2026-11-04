# Eight-hour RoboCasa monitoring and next-stage preparation

Owner: chat `01a1a717-4b4f-7173-88a6-4a3ffef3259e`.
Window: 2026-11-04 23:05 to 2026-11-05 07:05 Asia/Shanghai.
Human request: understand PPO/GRPO baselines and obtain effective policies;
monitor and evaluate the current meta-learners; once five task policies are
effective, retrain the joint learners and evaluate frozen-base steering on
ID, then five tasks in each RoboCasa benchmark category (atomic seen,
composite seen, composite unseen), with one through five episodes in context.

## Current work and ownership

- `job-967y1maorj80`: four joint three-task direct-action learners (AD, DPT,
  DIT, IC-IQL), learner pin `d71cb532`, under the existing cumulative 96
  GPU-hour budget. Preserve its immutable source, corpus, optimizer, sampler,
  RNG and validation stopping rules. Check progress at least every 20 minutes.
- `job-t6o21b4rl1x7`: six-policy OpenDrawer paired TEST512, owned by chat
  `01a1a633`. Read its artifacts and owner progress; do not duplicate or
  change the running evaluation. Stage 1 GRPO has DEV64 only, which cannot
  establish an independent TEST ranking against PPO.
- Existing source campaigns and explicit holds remain as recorded. Do not
  restart the held velocity/KV PPO sources or silently reinterpret the
  earlier selected-checkpoint admission family.

## Evaluation sequence

1. Independently inspect fit metrics, checkpoint freshness, finite values,
   completion/convergence receipts and the remaining budget. Freeze each
   method's validation-selected candidate once that method converges; dispatch
   its readout without waiting for the other methods.
2. Qualify six actual episodes per trial: episode 1 is the zero-context
   anchor; episodes 2–6 are queries following 1–5 completed episodes. Retain
   separate episode success, final success and cumulative ever-success.
   Never label best-of-six as a five-context query success rate. Every query
   uses its own preceding actual history with no deployment SGD.
3. Evaluate the current three-task direct-action baseline on fresh ID cases
   first, then its contracted OOD cases. Compare full memory with episode
   reset and original/final frozen anchors. Label direct-action results
   separately from steering through the frozen action expert.
4. Before a five-task fit, bind five effective source policies and all their
   chronological source histories. The pending CounterToCabinet/Faucet route
   still needs its recorded evidence gate, a concrete qualified recipe and
   a compute budget; the current instruction does not manufacture a numeric
   budget for two full source runs. Use the completed ablation to inform any
   proposed recipe, retaining the original contrasts.
5. Train/qualify learned-KV steering through the frozen original policy,
   then run ID before the 15-task transfer panel. Lock task membership and
   split definitions from the actual RoboCasa registry before outcomes.
   Source tasks are excluded from the 15 transfer tasks. No meta-training or
   checkpoint selection uses these evaluation outcomes.

## Resources and records

Use Baige allocated Linux resources for tests, model work and simulation.
macOS is limited to editing and static checks. Read-only SSH uses the local
development endpoint; credentials and routing remain outside this repository.
Before each new allocation recheck capacity, hardware, limits, storage and a
write probe, and run the actual in-allocation GPU probe. Charge failures,
qualification and release time. Fit/evaluation continuations may use the
remaining existing 96 GPU-hours. New work of eight GPU-hours or more needs
an explicit incremental budget under AGENTS.md. Do not split a larger campaign
into small allocations to bypass that threshold.

Write raw checks and cost estimates here. Edit only this chat's row in
`outputs/coordination.md`. Publish verified findings after independent audit.
Keep the chat active during the monitoring window; no scheduled automation
substitutes for following the jobs.
