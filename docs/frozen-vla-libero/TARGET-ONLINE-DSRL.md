# Fresh target-online DSRL learning-onset diagnostic

This baseline starts an independent pixel SAC actor, critic, optimizer, replay
and sampler on each predeclared LIBERO90 target: task0/state0/scene seed123 and
task1/state0/scene seed123. It uses the same frozen official OpenPI pi0.5 LIBERO40
motor and32D noise interface. It never restores the source SAC teacher.
All newly collected trajectories are target adaptation/evaluation data and have
`source_assembly_admission=false`.

The source collector CLI cannot perform this comparison: it deliberately admits
only source40 tasks and has no independently frozen target-query curve. The new
driver reuses its qualified `OnlineSourceSimulator` plus the unchanged
`OfficialSAC` learner, with the original64×64 camera/proprio8 interface,
batch256, UTD20 and replay strictly greater than500 chunks before the first
post-episode update. Actor/critic weights stay fixed throughout each episode.
Sparse chunk reward is minus one, zero on success; failures/timeouts bootstrap.
The original actor/critic/temperature rates and ten-critic architecture stay
unchanged. Native pi0.5 H10/execute10, fixed physical cases, actual-duration
chunk discounts and this short stopping budget are explicit departures from
the original pi0 LIBERO experiment.

The pinned authors' [`collect_traj`](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L193-L244)
collects unbounded tied32D Gaussian noise before any optimizer update and uses
`sample_actions` afterward. Its [`perform_control_eval`](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L300-L350)
uses independent full-horizon Gaussian noise for the initial motor evaluation,
then stochastic `sample_actions` after updates. This diagnostic preserves those
distinct collection/evaluation rules; it does not substitute `eval_actions`.

Each learner completes15 training episodes, with separate query episodes after
0/1/13/15 completed training episodes. Their declared worst-case training-action
ceilings are0/410/5,330/6,150. The learning curve reports actual primitive
steps, settling actions, query chunks, optimizer updates, success and training
returns at every episode and query. Early successes can shorten episodes, so
episode ceilings do not imply that warmup has cleared. If all episodes fail at
400 primitives, replay reaches520 chunks after episode13; SAC then performs800
updates, followed by two episodes using learned actor weights. This is a
learning-onset comparison and cannot establish converged ordinary DSRL.

Queries preserve all actor/critic/optimizer weights and replay, use one paired
query seed at every milestone, and restore training agent/motor/global RNGs
afterward. Query trajectories never enter replay. Every query retains its own
full recoverable learner checkpoint as well as its raw actual trajectory.

There is one motor, at most one owned simulator process, and two independent
learners used sequentially. The joint cap is16,000 simulator actions, with8,000
per case. This includes training, all eight queries, qualification settling,
uncertain RPCs and retries. The worst-case complete schedule uses12,300 training
actions,3,280 query actions and20 initial qualification actions, leaving400
retry actions. Only complete410-action episode reservations can start. A
45-minute allocation is a planning ceiling; the driver stops at complete
boundaries and resumes against the original source and cumulative caps.

Raw journals persist every RPC's public observations, proposed/executed latent
noise, native motor noise, motor actions, actual raw/chunk reward, primitive
duration, termination/truncation and reset identity. Unresolved calls are
charged at their maximum possible action count. The last complete episode
checkpoint includes policy/target critic/temperature, all optimizer states,
replay, sampler, global Python/NumPy, motor and agent RNGs, training/query
cursors and trace hashes. Partial attempts stay in the action journal and do
not become completed learning episodes after recovery.

After deployment and budget review, run inside an existing allocated GPU:

```bash
bash scripts/diagnostics/vla_target_online_dsrl.sh EXACT_COMMIT FRESH_RUN
# Exact same run, commit and16k reservation after a diagnosed interruption:
bash scripts/diagnostics/vla_target_online_dsrl.sh EXACT_COMMIT SAME_RUN --resume
```

The launcher reserves16k from the existing shared campaign pool before any
robot interaction. Parent campaign accounting must first retain this space
without expanding the authorized1M benchmark cap. This implementation does not
submit scheduler jobs or change any active source-training contract.
