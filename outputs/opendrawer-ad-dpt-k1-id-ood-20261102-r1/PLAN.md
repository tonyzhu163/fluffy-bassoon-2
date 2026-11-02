# Primary one-shot ID/OOD readout

Human clarification,2026-11-02: evaluate both ID and OOD,with one-shot as the
primary endpoint. This supplements the immutable convergence/baseline job
`job-8kb761dxfvtm` at68f2c5aa; it does not edit or stop that active run.

ID is the same50 fresh OpenDrawer cases already bound in the source plan.
OOD is50 fresh CloseFridge cases,held out from source PPO and student fitting,
following this project's existing OOD task definition. Native constructor
seeds880210000..880210049 are fixed before outcomes. CloseFridge's native
horizon is900,not OpenDrawer's750. OOD means held out from controller training;
the released foundation policy may already know this task.

Primary AD/full and DPT/full:one COMPLETE support episode with actual terminal
simulator feedback,then a separate query on the identical case/reset,with
frozen weights and memory carried across the boundary. Query motor draws
match the corresponding zero-shot query. Report support success separately.

Initial/final PPO and separately trained static students are context-blind
query controls. Their query policies cannot depend on support. Reuse their
matched K0 query outcomes; do not invent physical support episodes or call
them adaptive one-shot learners. Give full and reset/static comparisons the
same recorded support where interpreting memory interventions,and count
shared actual support cost once. Report actual interactions per method.

Preserve all existing ID K0 results and the existing first-eight K1 results.
After all four fits converge,run only missing ID K1 cases8..49 for AD/full
and DPT/full,and OOD K0 for all six policies plus OOD K1 for both full models.
This adds668 physical episodes:168 ID and500 OOD. Existing332 plus new668
equals1000 total physical readout episodes; conservative combined controls
are953000,including up to128 constructor controls per episode. This is fixed
cohort accounting,not a fitting cap. No teacher/student training on OOD.

Freeze selected checkpoint hashes/statuses and both PPO hashes in binding.json
only AFTER every fit reaches the original validation plateau. Keep original
training/data/optimizer/selection contracts unchanged. Evaluation outcomes
cannot change candidates,cohorts or hyperparameters. ID support/query draws
are identical to the source readout; OOD uses a distinct fixed private stream.

The additional driver may run concurrently with the original ID readout once
fits converge. It never waits holding GPUs for original ID results:release
its allocation on completing its own episodes,and aggregate on allocated
Linux CPUs when both phases have finished. Verify per-case initial-feature/
base-chunk pairing across six K0 policies and both K1 queries. Reuse existing
records only with matching checkpoint,case,noise and record hashes.

Execution:separate Baige job,4 nodes x2 allocated GPUs,32 CPUs/128GiB RAM and
16GiB shared memory per node. At most479 minutes per scheduler chunk,with
fresh attempt identities after diagnosis; preserve completed records and
partial journals. No aggregate GPU-hour cap substitutes for convergence.
Reserve64GiB output capacity,verify write access and actual GPU probes.
Each GPU's first retained OOD initializer episode verifies zero-residual parity
on all18 chunks before execution. Runtime motor/features/controller code stays
at original68f2c5aa; the new evaluation driver has a separately recorded Git pin.

Report separate ID and OOD tables:initial PPO,final PPO,AD/full K1,DPT/full K1,
static controls,plus full K0 and paired K1-minus-K0 comparisons. Include paired
wins/losses/exact tests,source and readout interactions,compute,and single-seed/
single-source-task limits. Complete only after both cohorts and the source
baseline phase are verified. Monitor every20 minutes through the same chat.
