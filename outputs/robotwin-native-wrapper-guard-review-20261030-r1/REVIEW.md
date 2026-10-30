# Future wrapper guard: static review and later CPU qualification

Reviewed `c166171b847ada9a20a9629a07d5a0100b517097` against active-base
`2fd2096906e52aabeaeb984f333fe1aaa6c4e196`. No new blocking static finding.
This review ran no tests, model/simulator work or allocations and changed no
active code, process or ledger.

The fixed-plan issue is addressed at both boundaries: `main` rejects a supplied
checksum other than `PLAN_SHA`, and `read_fixed_plan` independently requires the
same constant before reading the file. Other receipt checksums retain their
existing behavior; the declared plan and model/backend/controller bytes remain
unchanged.

The physical-worker parent-loss issue is addressed in the exec'd child before
Git, model, CUDA or simulator preparation. The parent supplies PID and Linux
start ticks. The child verifies PPID/start identity, sets `PR_SET_PDEATHSIG` to
uncatchable `SIGKILL`, verifies `PR_GET_PDEATHSIG`, then repeats the identity
check. The `/proc` field22 parsing is correct. Parent death before/during arming
and PID reuse therefore fail closed on the reviewed path. No multithreaded
parent `preexec_fn` is used. Guard evidence is written into the child intent.

The prepared tests cover changed-plan rejection, original-plan acceptance,
missing/reused parents, death during arming, signal readback and a real Linux
owned-parent SIGKILL while its worker occupies a separate session. These are
appropriate checks; they were not executed as part of this original review.

The later allocated-Linux CPU gate passed31focused tests in1.975seconds at
the exactc166 implementation, inside the123-test combined gate on Baige
`job-k1upj2ycy60q`. The independently retained joint receipt here has SHA256
`985d526de472d15c2c71e88f434e2b5ec7f71df68f6049ea6be8861f7bd065db`.
Owner evidence is `80943996d033155e17e242b045f82904c255b89f`; the exporter
recorded the matching qualification at
`f46101faa730e63ed5cea7e8617d434871958de7`. That status supersedes the former
pending gate. The job used0GPUs/4CPUs/16GiB, exit0, and no live source controls,
robot/VLA actions or production SGD. The owner's terminal scheduler receipt
reports Success after71seconds, adding zeroGPU-seconds.

The guard is now CPU-qualified for a future reviewed invocation. Fresh live
admission and deployment remain separate. The completed evaluation retains its
original2fd pin and is not repeated or changed by this software gate. This
guard stops the physical worker; it does not establish teardown of every helper
or the whole container. Unresolved physical intents still stay charged, and
actual cumulative scheduler spending remains required for any resume.

Findings and these limits were handed to the exporter, execution owner and
coordinator. No further implementation, test or GPU action was requested here.
