# Resume unfinished learners without reserving completed learners' GPUs

Prepared during the eight-hour watch. Do not submit while the original fit
allocation is live. Its coordinator may close after its final complete6600s
slice, before the scheduler's eight-hour ceiling; watch the actual release.

After scheduler terminal/release verification, inspect all four status files,
latest complete recoveries and original coordinator child-release records.
Allocate one actual RPBZZZ6 per unfinished method (16CPU/64GiB RAM/8GiB SHM per
GPU), with a120-minute scheduler ceiling. The newly reviewed wrapper delegates
the original TRAIN-WORKER and learner modules at unchanged d71cb532; the same
pooled assembly, exact6600-second invocation, global batch8, Adam, sampler,
RNG, validation candidates and plateau state are retained. There is no DDP or
batch/topology change inside an individual learner. Converged methods are
never restarted. Record the new scheduler identity and resource mapping in the
cumulative96GPU-hour ledger before further work.

CONTINUE-FITS.py is a narrow adaptation of the original coordinator: dynamic
method/GPU membership, separate wrapper and immutable learner identities,
mandatory complete recovery and released-original guards. The training
arguments are identical. CONTINUE-FITS.sh performs the real allocation probe,
checks CPU/RAM/SHM/storage and cumulative reservations, and invokes one full
slice. Another slice needs a fresh output attempt and another release check;
there is no unattended retry or stopped-run replacement.

At launch, review current main against d71cb532's trainer, value learner,
policy, decoder, checkpoint and sampler code. Preserve the original learner
identity even when only the allocator/metadata changed. If relevant learner
code changed, diagnose and explicitly qualify a migration rather than editing
an active checkout or rewriting its saved contract.

This is continuation of the approved three-task96GPU-hour campaign. It does
not authorize the pending70GPU-hour source proposal or the later five-task fit.
