# Stage 0 recovery after the human-directed image-check repair

Owner chat01a19b58. This is the sole fresh recovery of Stage0; no duplicate
screen, renderer qualification or PPO training is launched.

The user subsequently requested fixing the overly strict image check and
resuming in chat01a19b74 (trusted user messages read via read_thread). Shared
repairf804ce6a keeps exact physical/model/camera/public-state and input-format
checks, but records RGB differences diagnostically. It has its own allocated
Linux test and actual failed-reset fixture evidence. This supersedes the
unlaunched renderer-threshold qualification proposal. Do not edit the old
ca5037da execution worktree or reclassify its failed reset.

The original screen job failed17:00:36 Shanghai; all four pods ended by
17:00:37. Its conservative scheduler lifetime charge is2924GPU-seconds,
leaving40276 of the original43200. This fresh job may request four1GPU/
32CPU/128GiB/16GiBSHM instances for at most160 minutes:38400GPU-seconds,
giving an aggregate upper bound41324. No larger budget or new scientific
scope. No PPO optimizer. Supervised direct-head qualification remains allowed.

Reuse the exact original selected20 failure cases per task, all136 complete
native anchor episodes, their complete chunk records, and fixed KV/velocity
scale-selection results. Independently audit the209 complete native-resampling
episodes' truth/video/case/reset evidence and source-index schedule. They lack
per-chunk records because the interrupted bank was saved only at its end;
disclose this limitation, never fabricate those records, and never use these
native screen rows as AD/DPT training histories. Their valid outcomes may
remain diagnostic headroom evidence. Carry them by immutable references and
collect only the431 missing native-resampling episodes, with the same original
case/draw seeds. Preserve unresolved old calls in accounting. No outcome-based
replacement, new seed or changed failure selection.

All other scientific arms retain the original contract in
../robocasa-interface-screen-20261102-r1/EXECUTION.md. The new collector saves
case/motor/control plans before dispatch and complete chunk records after
every completed episode. It checkpoints successful sibling episodes before
propagating another lane's failure. New reset snapshots are compared with
copies of the original canonical snapshots; original source files stay read-only.
New episode IDs include the recovery identity. Direct-head source records and
checkpoint gates, selector costs, original observations and frozen motor remain
unchanged. The actual new GPUs require probes, distinct identities and real
advancing episodes. Each task publishes combined carried/new outcomes, and
new interactions are charged separately from reused data.

Test the core complete-episode checkpoint hook, physical pairing repair and
actual saved fixtures on allocated Linux CPU before launch. Static checks only
on macOS. Pin latest reviewed main in a new detached execution checkout.
Record terminal old costs and all fresh costs, then monitor every20 minutes.
