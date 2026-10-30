# Approved LIBERO bulk collection

Human approved32additionalBaigeGPUhours on2026-10-30: eight SAC workers
up to2h each, full and independently trained reset PPO up to6h each, and
frozen source-local validation up to4h. The existing1M action ceiling and
approximately630GB storage allowance remain. RoboTwin is a separate pending
request. See [approval](approval.json).

Completed source slots are preserved and skipped through the canonical
ledger. Fresh run identities preserve older prepared worktrees. Actual
allocation probes/tests precede new model interactions. Full recoveries and
all physical costs remain authoritative; this approval is not a scientific
performance claim.

## Running allocations

All10freshallocations are on exactsource531e0f838a0063abfa28c4b3940cd24dfaf6edbf,
with10distinctactualGPUs. EightSACworkers plus sharedPPOfull and separately
trainedepisode-reset each passed303Linuxchecks (2JAXprobe-environment skips).
See [job/compute ledger](compute-ledger.json) for IDs, source pins and limits.

Saved-checkpoint audit at2026-10-30 11:21Shanghai: SAC has183completeK4training
casegroups and740committed trainingepisodes, including in-progresscases.
The first newSpatial08stream completed16cases/64episodes/764chunks/5,300SAC
updates/7,976actions; its full56MBrecovery was inspected. PPOfull/reset
have30/29iterations and120/116completeonpolicyepisodes; their optimizer,
sampler/taskcursor/motor/globalRNGrecoveries were inspected.
[Runtime/checkpoint snapshot](runtime-qualification.json). These are actual
execution counts, not evidence of robotperformanceimprovement. Physical
executionjournals, including qualification and liveRPCceilings, are the
authoritative actioncharges; learnercounters alone omit those costs.

Collection proceeds in parallel. SAC and PPO update during collection to
produce real learninghistories/onpolicydata; offlineAD/DPTtraining follows
from the frozen source corpus. Both production assembly/milestone and
frozenPPOtargetevaluation bindings are being prepared while jobs run.
Frozen sourcevalidation remains unlaunched until both172kPPOcheckpoints are
ready; its4hour/56kaction hold remains intact. The continuation monitor is
`continue-approved-libero-source-collection`,20minutes, quiet whilehealthy.
RoboTwin is also collecting under its separate owner and humanauthorization.
