# Grouped reset/replay passed; intermediate FSDP representation guard failed

Job`job-l218ew6o4lfw`, source`92531a7a`, completed all32scene groups×8replicas:
requested/effective seeds, prompts, initial RGB/state and exposed articulation
state were exact within each group. Raw replicas/differences were persisted.
All four first-replay logprob errors passed. Each rank's first backward has
190finite, nonzero policy gradients and zero critic-gradient tensors.

The new harness then incorrectly required every frozen parameter's grad to be
None after the first of eight accumulation microbatches. The installed FSDP
use_orig_params=True implementation assigns unsharded flat-gradient views to
all original parameters during no_sync; only its synchronized sharded-gradient
path explicitly clears requires_grad=False views. This source finding identifies
an invalid representation assumption, not actual frozen-gradient leakage.
The failed parameter's numerical view was not persisted before the assertion,
so its exact values remain unknown. No tolerance is relaxed or gate removed.

No GRPO optimizer update or step1 checkpoint/recovery completed. The initial
full checkpoint, original canonical pre-SGD batch/RNG, actual grouped reset
records and all prior evidence remain immutable. A targeted actor-only replay
will record every frozen view and optimizer membership before assertions,
require finite exact-zero/empty intermediate views, None at the synchronized
boundary, unchanged frozen hashes after a real update, and independent recovery.
The full grouped-runtime qualifier still needs a fresh successful run.

Created03:01:41Shanghai, released03:19:11. Scheduler1030sec×4=4120GPU-sec;
creation-to-release1050sec×4=4200; worker1001sec×4=4004. Conservative requested
motor/setup charges remain in this attempt's ledger; no measured delivered
action/physics-substep count is claimed.
