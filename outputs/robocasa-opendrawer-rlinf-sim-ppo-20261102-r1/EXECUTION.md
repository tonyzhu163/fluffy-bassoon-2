# Native Flow-SDE PPO execution binding

Human-delegated execution owner: parent chat01a19b74, subagent flow_sde_ppo.
Stage0 interface recovery remains with its existing owner. This run does not
depend on that screen and does not launch Flora or residual PPO.

Use one two-GPU Baige node, 64CPU/256GiB RAM/32GiB shared memory and a479-minute
scheduler deadline, within the existing aggregate40 experiment-GPU authority.
This is native RLinf PPO on clean patched runtime8a1233c3. The action expert,
fresh value head, learner, optimizer, GAE and Flow-SDE sampling remain native.

Qualification is iteration1 of the same18-iteration learner: it is checkpointed
and must pass finite-loss,16-optimizer-step, frozen-VLM and expected batch checks
before iteration2. It is not an extra rollout iteration. Fixed before64 and
after64 panels plus18x128x400 train controls total972,800 policy controls.
Constructor/reset controls and failures are separately observed and included in
the same1M-action escrow; never silently restart18 new iterations after a
qualification failure. Save full RLinf actor/Adam/LR/RNG and rollout/environment
boundary sidecars at0,1,6,12,18. Retain every actual chronological rollout and
sampling record, including producing-policy IDs, sampled chains, Gaussian
parameters, old log probabilities, raw rewards and masks.

## Source audit corrections and reset identity

The native relative reward is a signed difference of successive success flags.
That can cancel a transient success inside H50 before chunk-level aggregation.
The explicit human-selected first-success contract therefore uses a positive
one-time latch (+1 at the first native success,0 thereafter), preserving the
native first-termination loss mask and full400-control accounting horizon.
Core numerical qualification checks transient success and no later negative
reward. This corrects the planned reward semantics; it does not add shaping.

Native constructors are created only for actual episode banks. Each TRAIN
episode gets an independent fresh native constructor with seed
210100+iteration*128+round*32+global_env_index. Evaluation always uses the same64
fresh native seeds410100..410163 before and after. Python/NumPy are seeded inside
each child before construction, so decorative scene state is reproducible.
This explicit fresh-case schedule makes whole-iteration recovery complete
without attempting to pickle live MuJoCo objects. Environment sampler/RNG,
trajectory counter and episode-round cursor are retained in sidecars.

Before/after resets require exact physical scene, camera geometry, public state
and task identity. RGB differences are diagnostic under the human-approved
policy in f804ce6a; actual policy pixels stay unchanged. Per-case native truth
and observed controller/physics counts are saved at each400-control completion.
No comparison is inferred from a queued/running scheduler badge.

## Launch and admission

Deploy a verified main commit through Git into a fresh detached worktree. Bind
TASK_META, TASK_CODE_PIN, TASK_ATTEMPT and TASK_OUTPUT, then run COMMAND.sh.
The job must pass an actual two-GPU probe before model loading. Record the
allocation, exact runtime/model/norm/source pins and storage write/capacity
checks. The before-panel success_once must fall in5–70%; a result outside that
range stops for the predeclared human task decision. No automatic task swap.
After qualification verify actual advancing updates and complete artifacts,
then monitor every20minutes. Preserve partials/costs before any fresh attempt.
