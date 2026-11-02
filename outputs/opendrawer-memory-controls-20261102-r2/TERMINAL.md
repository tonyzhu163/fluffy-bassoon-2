# Matched controls interrupted after 1052 verified collected endpoints

Baige Pro6K / 算法Z组 job `job-dxxkwgvdk4hy` failed at **00:43:33
Shanghai, November 3, 2026**. All eight allocations are terminal: the master
failed, three siblings were stopped, and four siblings succeeded. The pinned
source `1b9291bdc5729490cf0bf34f00c55a777e8bfe23` and all prior worktrees,
bindings, completed episodes and interrupted artifacts remain preserved.

At **01:10:36 Shanghai**, an allocated-Linux CPU verification passed for
**1052/1100** collected endpoints: **785** from r2 and all **267** carried
from r1. It checked source/candidate identity, original cases and motor seeds,
record/truth hashes, canonical and current snapshots, actual native pairing,
750/900-control record lengths, terminal reward alignment and reset/window64
cache lengths. All eight original allocation probes have distinct GPU UUIDs.

| Collected control endpoints | Verified | Planned |
| --- | ---: | ---: |
| AD reset ID / OOD | 200 / 200 | 200 / 200 |
| AD window64 ID / OOD | 50 / 100 | 50 / 100 |
| DPT reset ID / OOD | 163 / 200 | 200 / 200 |
| DPT window64 ID / OOD | 39 / 100 | 50 / 100 |

**48 DPT/ID episodes remain absent:** 37 reset and 11 window64. Their exact
case/attempt keys are in [TERMINAL-PARTIAL.json](TERMINAL-PARTIAL.json).
The local manifest matches the remote bytes exactly, SHA256
`df7b5f4732fde766737f962250373903ee853f3f5819c83e089268c11a935efa`.
The final 2000-control comparison has not been produced. Do not interpret
partial curves as the complete matched experiment or claim a memory benefit.

## Diagnosis and prepared correction

Rank 0 finished its assigned simulator work and wrote its successful worker
marker. The parent then called `BASE['wait_for_nodes']` in `CONTROL.py`.
`BASE` loads an immutable earlier readout which predates that helper, so the
master raised **`KeyError: 'wait_for_nodes'`**. Baige marked the job failed
and stopped the three unfinished siblings. This failure occurred at the
cross-node barrier; it does not invalidate the completed episodes that passed
independent verification.

The canonical future driver now loads the helper from its own committed
`READOUT.py`, as the actual-attempt driver already does, and passes the actual
node count. The terminated r2 worktree is unchanged. Allocated-Linux CPU
checks confirmed the original missing key and that the selected helper
accepts eight successful node terminals, rejects a failed peer, and rejects
a missing peer at an expired deadline. The helper's source matches current
main exactly. [Run-local check](BARRIER-CHECK.json).

The [scheduler closeout](SCHEDULER-CLOSEOUT.json) conservatively charges r2
45600 GPU-seconds (8 GPUs × 95 minutes, above the longest 94m20s instance
lifetime), plus r1's 17280: **62880**, below the original 86400 ceiling.
CPU verification adds no GPU collection.

No automatic retry was launched. A further decision is required before a
new output identity collects the remaining 48 episodes with the corrected
barrier. Reuse every verified endpoint, replay carried window64 prefixes,
then run the committed full summary only after all 1100 collected / 2000
control endpoints and all worker completion markers are present. The failed
job's heartbeat is removed; this chat remains open.
