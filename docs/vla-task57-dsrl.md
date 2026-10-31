# Task57 ordinary DSRL with frozen pi0.5

`vla_target_online_dsrl_task57` fits a fresh authors' PixelSAC noise controller
on the authors' predeclared LIBERO90 task57 (cream cheese to tray). The entire
official OpenPI40 `pi05_libero` motor is frozen. This is a pi0.5 adaptation of
the released pi0 recipe, with native H10/execute10 and actual-duration discount;
it is not a literal pi0 H50/execute20 reproduction.

The actor consumes single-camera64×64 RGB and8D proprioception. Its32D action
is repeated across10 motor slots. Canonical SAC uses batch256, UTD20,10 critics,
LR1e-4/3e-4/3e-4, and Gaussian warmup until the first update with replay>500
query records. Actor weights remain fixed during each complete episode. Rewards
are−1/query and0 on success; timeout bootstraps; discount is .999^actual steps.

Training cycles states0..39 without outcome selection. Query panels use fixed
states40..49 and the same per-case seed bank at every milestone. Initial/final
native panels use independent H10×32 Gaussian noise via the explicit noise
path; learned panels use stochastic SAC. Query records never enter replay or
source40 meta pretraining. All task57 records have role`target-online-baseline`.

The nominal stop is500000 optimizer updates after a whole episode/update burst,
with at most780-update overshoot. Ten-query panels run at the first completed
burst crossing each10000-update threshold and report actual optimizer counts.
No update is truncated to manufacture an exact threshold. This one-seed curve
does not establish convergence or reproduce the paper's three-seed confidence.

The whole phase has one SHA-bound grant, <=36000 cumulative GPU-seconds,
<=1000000 actual simulator actions and <=64GiB output storage. All setup,
qualification, queries, settling, failures and retries share those ceilings.
Admission reserves a whole410-action episode, time margin, and at least4GiB
storage (increased when measured slot/trace sizes require it). Storage exhaustion
stops before the next episode, preserves data, and does not imply method failure.

Two alternating atomic checkpoints retain full SAC/Adam/replay/sampler/global
and typed motor RNG. An atomic JSON cursor binds the current full state's SHA.
Immutable milestone hardlinks preserve selected full states. Query completion
updates only the cursor because it restores all training RNG/model state.
Raw traces and intent/outcome journals retain actual observations,32D/H10 noise,
executed actions, rewards, durations, valid masks and initial-state hashes.
Unresolved physical intents remain conservatively charged on resume. Before
and after each loaded allocation the whole motor parameter hash must match.

Inside an approved allocation, after deploying the exact clean integrated pin:

```sh
bash scripts/diagnostics/vla_target_online_dsrl_task57.sh \
  SOURCE RUN GRANT_JSON GRANT_SHA256 7140 --pause-after-updates 10000
# Same source, run and grant; next slice continues the same actor and costs:
bash scripts/diagnostics/vla_target_online_dsrl_task57.sh \
  SOURCE RUN GRANT_JSON GRANT_SHA256 7140 --resume
```

The run output must be fresh and distinct from the grant/plan directory. The
pause and slice length change execution only, never scientific phase limits.
The launcher checks the actual allocated GPU and runs only the new glue/recovery
tests before model execution. It never requests an allocation itself. Setup-only
failure can resume with its old cost journal; missing scientific recovery after
training/query actions is rejected. Protected prior onset/six/long runs are not
modified or reused.
