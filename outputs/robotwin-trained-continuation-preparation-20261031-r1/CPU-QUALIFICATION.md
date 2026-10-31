# Allocated Linux CPU qualification

Baige job `job-q6dhvcewwwx1` passed both software phases at exact clean source
pins. No GPU, simulator, robot actions or source-process controls were used.
The allocation receipt confirms hostname `job-q6dhvcewwwx1-master-0`, zero CUDA
devices, a four-CPU cgroup and 16 GiB memory. Test driver source was
`5bbddf7eaff3d8ff58c50e5fb359c18fbbb5f1d8`.

| Phase | Exact code pin | Passed | Failed/errors/skipped | Phase wall time |
| --- | --- | ---: | ---: | ---: |
| RoboTwin continuation and paired launcher | `0c16c2b794f8de5b78603206917e72195083c265` | 67 | 0/0/0 | 74.828 s |
| LIBERO cached scalar adapter fake tests | `26b55eae2a5ce6545cdc97b3a875cc6e3a7b46fa` | 43 | 0/0/0 | 0.468 s |

The RoboTwin phase includes 14 continuation, 17 plan, 16 preflight, 14 runtime,
three audit and three real owned-process spawn regression cases. The latter
exercise TERM, INT and a registration-callback failure between Popen and
registration, including descendant cleanup, an unrelated surviving session,
and unblocked child signals. These are CPU fixtures, not native robot episodes.

The combined driver ran 02:18:31–02:20:23 on 31 October 2026, Asia/Shanghai,
111.203 seconds elapsed. This is driver time, not an asserted terminal scheduler
charge. The owner separately retains scheduler terminal accounting.

Root downloaded the summary, actual allocation receipt, both JUnit XML files
and test logs. Static local readback verified the bound XML/allocation hashes,
actual cases and zero unsuccessful cases, exact clean before/after sources,
all three spawn cases and the allocated CPU limits. See
`linux-cpu-gate/LOCAL-READBACK.json`; the local readback executes no tests.

This qualifies the tested continuation software for the owner's original
100-round operational handoff. It is not evidence that the continuous GPU run
has launched or completed, nor a native performance, convergence or adaptation
result. Keep runtime570, global batch6 and all scientific contracts unchanged.
Use latest owner-issued final manifests and canonical heads, not historical
examples in the preparation review. The execution owner controls the future
allocation and actual-charge reconciliation.

The 43 LIBERO cases qualify only the adapter at26b55ea on CPU fakes. They do
not qualify the later recorded-input CLI, its 56 prepared tests, an actual GPU
comparison or source adoption. The optional LIBERO GPU gate remains deferred
and unfunded.
