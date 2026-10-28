# CTM T1 PPO optimization qualification

User requested porting the original CTM RL PPO optimizations, measured
per-rollout/per-update times, CTM default one internal iteration, and an
actual ML10 comparison against RL², ECET and AMAGO. This admission measures
the optimized CTM path before budgeting that multi-method comparison.

Reviewed latest main: `c7042e23a96c40d8c384625280390b1e19e7487b`; merged
into this task detached worktree. Upstream CTM RL source:
`SakanaAI/continuous-thought-machines@4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`.

Use a new immutable Git commit and output identity. Baige, one exclusive
RPBZZZ6, 16 CPUs/64 GiB RAM, at most 25 scheduler minutes. Preserve NVIDIA
visibility. Run the allocated GPU probe, targeted CPU/CUDA suites and real
ML1/ML10 wrapper checks before timing. Do not run tests on dev or macOS.

Compare one update each of serial-head full-trial PPO, batched-head full-trial
PPO, and cached-state 100-step TBPTT. Use identical CTM T1/width84/depth8
weights and a real ML10 10-lane x 5 x 500 batch. Five optimizer steps, one
PPO epoch and 25,000 training transitions per timing. Measure two paired
collections with/without state caching; require identical recorded data.
All replay modes must reconstruct frozen likelihoods/values within 2e-5.
Record peak CUDA allocation, achieved samples and optimizer work.

Full-trial batching preserves its objective. Cached chunks preserve forward
history and full-trial GAE, but truncate recurrent gradients and change
minibatch membership; record this as a separate protocol. No convergence
claim follows from these timings. The baseline comparison will use actual
RL², ECET and AMAGO implementations, not relabel GRU/XL as all three methods.
