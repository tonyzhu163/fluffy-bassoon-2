# Future wrapper guard: static review

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
appropriate pending checks, not executed evidence from this review.

Qualification remains pending the focused allocated-Linux gate in the next
reviewed invocation. The current evaluation stays on its original pin. This
guard stops the physical worker; it does not establish teardown of every helper
or the whole container. Unresolved physical intents still stay charged, and
actual cumulative scheduler spending remains required for any resume.

Findings and these limits were handed to the exporter, execution owner and
coordinator. No further implementation, test or GPU action was requested here.
