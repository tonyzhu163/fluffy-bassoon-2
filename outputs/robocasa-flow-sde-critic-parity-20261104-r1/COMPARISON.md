# Three-update critic comparison, declared before launch

The saved-batch audit found one active initial-policy decision with a value
gap above .2 and a suppressed critic gradient. This admits PLAN phase 3 even
though the 192 later-policy decisions show no initial clipping effect.

Compare the historical ten-suffix rollout mean with the qualified selected
suffix value on Drawer→Counter, **three complete PPO iterations per arm**.
Both start from the original q1 policy-0 full DCP checkpoint, Adam/scheduler
state, and all 18 actor/rollout/environment RNG/cursor sidecars. Restore into
a private shared checkpoint copy; never edit the original source or runtime.
The original policy-0 64-case before panel (32/64 success) is reused, with its
canonical states and episode hashes verified. Use TRAIN seeds 2300000–2300383
and the original 64 after-evaluation cases 630100–630163.

Hold the runtime at `0988b5f9` in both arms. Review current main before launch;
the older runtime is intentional here to hold every learner/environment
change fixed. Baseline uses clean RLinf `8a1233c3`; fixed uses a new detached
Git commit containing only `rlinf-8a1233-selected-suffix-critic.patch`.
Use the same BF16 model, frozen VLM, 8 env/rollout ranks, 2 actor ranks, 128
episodes per collection, H600/H50, global batch256, microbatch16, four epochs,
24 Adam steps per collection (72 per arm), learning rates and loss weights.
The instrumentation records unchanged-weight values before each iteration's
first backward; the fixed arm must pass value and native likelihood parity.
No mismatch assertion is bypassed in the fixed arm. Baseline value mismatch
is an explicitly observed historical-method control.

Run arms sequentially in one exclusive Baige Pro6K allocation: 8 GPUs,
168 CPU,1280GiB RAM,32GiB SHM, at most42 scheduler minutes, at most40 process
minutes total, and at most19 process minutes per arm. Automatic retry is off.
Charge all eight GPUs during initialization, cleanup and both arms. Including
the prior393 conservative GPU-seconds, the scheduler maximum is20553 seconds
(5.71GPUh), below this validation's21600-second cap. Check actual queue/device,
storage and the existing Linux CPU/device qualification before launch.

Per-arm conservative action ceiling372736, aggregate745472, includes the
contract's unused before-panel allowance and reset reserve. Expected fresh
controls are230400 TRAIN and38400 after-eval per arm (537600 combined), plus
at most114688 reset-reserve controls combined. Separate accounting roots
prevent one arm from consuming the other's ceiling. No fresh before panel.

Stop after both third updates and fixed after panels; fail on parity, finite
loss/frozen-VLM checks, budget/storage guards, missing restore state, or timeout.
Keep complete resumable final checkpoints and failed-attempt evidence. No
automatic retry or extension follows failure. Monitor at least every20min.

Read out critic/value-clip metrics and losses across all three updates,
TRAIN first-success counts, final paired64-case success, and initial RNG,
reset/state and first-action agreement. Report any simulator divergence;
known repeated-evaluation variability prevents treating small success
differences as causal proof. Three updates and one seed do not establish
convergence, cure the failed teacher gates, or replace either10M result.
