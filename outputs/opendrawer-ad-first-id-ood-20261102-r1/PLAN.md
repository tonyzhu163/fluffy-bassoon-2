# Independent AD-first ID/OOD readout

**Attempt ad-first-r1 failed its exact pairing guard.**32 complete episodes and5 unresolved calls are retained;31,000 controls are conservatively charged. No paired result is admitted. See INCIDENT.json and the separate reset-pairing diagnosis. Training and checkpoint selection remain unchanged.

Human steering,2026-11-02: converged models should evaluate separately; fix the
pipeline. The old all-four-model barrier delayed AD unnecessarily. It is
removed from new dispatch. The active legacy training job stays immutable
while DPT/static finishes. Its redundant evaluation is superseded by these
independent readouts; after all fits finish,preserve full recoveries and verify
ownership before closing that owned legacy allocation. Retain/account any
legacy evaluation records produced before closure.

Dispatch only ready models. This first phase selects initial PPO,final PPO,
AD/full and AD/static,whose checkpoints are already frozen by their own
validation plateau. DPT readiness is not inspected and cannot gate AD.
DPT/full and DPT/static get separate phases when each is ready; only the
overall summary waits for every requested method. Never use evaluation
results to change training,checkpoint selection or the locked cohorts.

Use the generic READOUT.py and COMMAND.sh in
`outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1`,configured by:
`TASK_READOUT_OUTPUT=/mowenjie/fluffy-bassoon-2/outputs/opendrawer-ad-first-id-ood-20261102-r1`,
`TASK_READOUT_METHODS=["ppo_initial","ppo_final","ad_full","ad_static"]`,
`TASK_READOUT_FRESH_ID=1`.

Each phase's binding admits ONLY its selected models and records their exact
best-validation checkpoint/status hashes. Runtime motor/features/controllers
remain at68f2c5aa; the independently pinned driver is the latest reviewed main.
Per-phase results and paired comparisons publish without any other learner.

Same fixed ID50 OpenDrawer and OOD50 CloseFridge cases/noise plans as the prior
one-shot plan. AD/full gets K0 plus one complete support and a separate same-case
K1 query,actual terminal feedback,frozen weights. PPO and AD/static are
context-blind K0 query controls. Report support success separately,and K1-K0
memory comparison on paired query innovations. No imaginary support actions.

This ready-only AD phase has600 physical episodes,conservative571800 all-in
controls. Subsequent DPT/full has300 episodes and DPT/static100:the intended
combined independent readout remains1000 episodes/953000 conservative controls,
plus any separately retained legacy spill,all charged explicitly. A provisional
six-GPU-hour/200k-action cap does not apply. No new PPO/source training.

Use a separate Baige4x2-GPU evaluation allocation,32CPU/128GiB RAM/16GiB shared
memory per node,at most479 minutes per scheduler chunk. Actual GPU probes and
first retained native episodes qualify the unchanged runtime. Preserve all
complete/partial records and diagnose failures before a fresh attempt. Release
each phase's GPUs when its own work completes. Monitor every20 minutes.
