# Source SAC continuation qualification

Exact source e7900a75a089db7e0cbf56de8c2e8f2beea90f02; original main reviewed
acca706 unchanged on2026-10-31 at02:05 Shanghai. Plan file SHA
09835ec9b620b136c79ca58ea2429dd5dfb722ddc64042e42f79763eff4c6720,
content SHA55a7b224e9f0dd5175d88f13b5876f066505724b360cf781a7fca952a045634a.
Only Spatial00/train is launchable:7,892 inherited actions plus at most5,000
new, under a cumulative1,000-GPU-second qualification funded from original32h.

First allocation job-g8ghgrpu8334, submitted02:05:46, failed preflight before
any tests, model, simulator reset or new actions. Missing CUDA_VISIBLE_DEVICES
caused the existing GPU probe to reject the runtime. Clock retains76s, exit1;
no experiment was created. All failed-attempt files are retained. This says
nothing about SAC learning or full-state migration.

Retry is an exact-source explicit resume into a new immutable attempt directory,
with the qualified fallback CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}.
This retains a nonempty scheduler namespace and only uses0 when unset in the
exclusive one-GPU container. Actual allocated probe remains mandatory. The
retry slice is864s plus60s allowance:76+864+60=1,000. It has a14-minute scheduler
limit and no automatic retries. No fresh1,000s grant, source contract rewrite
or active-worktree edit occurs.

The second allocation job-68ino91bxbua passed the real GPU probe and stopped
on a test-fixture omission:37 passed,2 skipped,1 failed. The fixture left out
selected.slot before its intended rejection assertions; runtime selections
include slot. It used70s and zero source actions. Original r1 clock retains
146s across both failures; no experiment was created.

Fresh repaired code cf49bfac65a98cc7c4cef7fb35df4a6a18e20c0b includes the
fixture fix and in-script GPU fallback. A new r2 output/clock has worker854s,
sharing the canonical qualification ledger with old146s. The launch slice is
700s plus60s scheduler allowance under that854s worker hold, with a12-minute
scheduler limit; the unused94s is not automatically spent. The aggregate
1,000s bound is unchanged. qualification-r2.json binds the predecessor plan
hashes and unchanged48 parents/slots/costs; only the execution source is rebound.
No scientific collector, optimizer or case schedule change is introduced.
