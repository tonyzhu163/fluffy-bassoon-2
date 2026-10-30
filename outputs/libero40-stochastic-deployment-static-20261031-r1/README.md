Static finding,2026-10-31: the examined OpenPI source40 AD/DPT robot evaluation path already samples stochastically. Deterministic-mean deployment does not explain the observed gap in those readouts. No new mean/sample diagnostic was implemented, funded or launched; the1,180 unassigned old-pool seconds remain untouched.

Source evidence, reviewed at root `f44d032a9228af59f9e3ea14639a359296366b49`:

- `src/fluffy_bassoon/meta_rl/vla_history.py:171`: `log_std` is a learned global per-channel parameter. Lines202–204 use `std=exp(clamp(log_std,-5,2))` and return `mean + std*randn` unless explicitly deterministic.
- `vla_history.py:186` and`:284`: both policy `step` and deployment-controller `act` default to `deterministic=False`. Lines297–298 forward that flag and the episode generator to the policy. Calling `.eval()` on the module does not remove its explicit random draw.
- `vla_source40_evaluate.py:200–201` constructs `FrozenHistoryController` without changing that default; lines281/284 call `controller.act(...)` without a deterministic override. The actual sampled32D proposal is then clipped and repeated across H10 at lines287–288.
- `vla_common_runtime_evaluate.py:213–217` uses that same controller in the shared-motor wrapper. `vla_source40_two_goal_evaluate.py:102–103` forwards `act` unchanged through `RecordingController`. Thus the common-runtime/firstwave chain preserves stochastic deployment.
- `vla_history.py:220–224` uses `deterministic=True` to obtain the mean for Gaussian NLL during offline fitting, then evaluates the target likelihood with the learned global std. That training call is not the robot deployment call.
- `vla_distribution_control.py:33–38` also obtains the mean deterministically, then explicitly draws `epsilon` and applies `mean + std*epsilon`. Its prior distribution intervention is not a hidden mean-only deployment path.

Historical source cross-check: deployed all90 firstwave pin `d095a2a4e66a0b79d03c9e823a183c19452d0c10` has the same default-false `step`/`act` and explicit Gaussian draw at lines186/203/284. The earlier qualified common-runtime pin `f5552d6` also calls `controller.act` without overriding determinism (evaluation lines279/282). This is not a sampling behavior added after the robot results.

Scope: code-path evidence for these source40 AD/DPT evaluators, not a new rollout, empirical dispersion estimate or performance claim. The fixed100 target profile remains qualified by its existing48 allocated CPU tests; actual target GPU execution remains separate and unfunded. Source collection is not gated by this closed diagnostic branch.
