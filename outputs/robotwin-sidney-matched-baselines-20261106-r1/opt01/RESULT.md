# Native optimizer representation diagnostic

Baige Pro6K job `job-6npadxhz5bln` completed at 04:34:18 Shanghai on Nov 6, from immutable `deefe9f559db0539fceedb35d159f15223db487d`. Four GPUs, no environment workers, no SGD. Native save and an independent fresh-process load reproduced the earlier raw optimizer digest mismatch on every rank.

All named parameter layouts, group ordering and typed hyperparameters match. Every locally owned Adam step and moment matches exactly in dtype, shape and tensor bytes; CPU/GPU placement changed through native loading. State entries fell from 190 on each rank to 113/41/41/118, precisely the locally nonempty parameter counts. The dropped 77/149/149/72 entries belong exclusively to zero-length unowned original parameters: finite zero steps and empty moments. Model weights, scheduler, scaler, RNG and frozen VLM hashes also match on all four ranks.

The installed Torch `_optim_utils.py:389` returns `{}` when an original parameter is outside the local shard. `_flat_param.py:2189-2210` assigns those parameters no synchronized gradient. The measured difference is representational. Raw hashes and all sixteen full named descriptors are retained. `SEMANTIC-COMPARISON.json` records file hashes and the exact comparison.

The new comparison preserves all parameter names/layouts, group metadata and all owned state. It accepts omission only for validated zero-step/empty-moment entries on zero-length unowned parameters. Unknown or nonzero discarded state fails. No numeric tolerance, dtype conversion, optimizer mutation or weight-only recovery is introduced.

This zero-update diagnostic does **not** qualify trained optimizer restoration, frozen gradients, full twelve-worker runtime restoration, or the science pair. Next is one native GRPO update on the immutable qual02 canonical rollout from fresh original Sidney, with full legacy initializer signature identity checked before replay, followed by independent post-update model/Adam/scheduler/scaler/RNG restore. Full simulator/group/case-ordinal qualification follows separately.

Scheduler charge: 924 GPU-seconds; conservative creation-to-release: 1,000; worker: 868. Closed cumulative task actual: 64,719 GPU-seconds (17.9775 GPU-hours). Conservative future gradient + full qualifier + bounded science pair worst case: 161,674 GPU-seconds (44.90944 hours), below the human 50-hour cap. Science admission remains false.
