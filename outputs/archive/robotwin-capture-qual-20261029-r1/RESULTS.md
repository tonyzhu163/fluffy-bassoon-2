# Final-camera capture qualified on two source scenes

Baige `job-lspkxidy3haj`, source `8114dd23e872d673ade6ea391b973c05a0dc6eb1`,
completed with exit0 at23:14:13 Shanghai,2026-10-29. Scheduler duration4m31s;
driver230.04seconds. Allocation/CUDA checks and **57 focused Linux tests**
passed (4.58seconds).

Both repeated native reference traces matched exactly. The opt-in final-camera
collector then matched **every public RGB/state/instruction/feature hash,
decoded joint-action hash, reward, terminal flag and executed count**, including
terminal observations, in both scenes. One scene timed out at400actions/13chunks;
the other succeeded after117actions/4chunks, covering partial final-chunk
execution. All modes shared the exact saved Gaussian noise bank and unchanged
frozen motor. See [qualification receipt](qualification/qualification.json)
and the six per-episode raw trace receipts in the same folder.

Paired reference episode time41.785seconds versus19.180seconds with deferred
camera reads: **2.1785x speedup**, about26.96executed actions/second across the
two candidate scenes. This does not establish the same gain on all six source
or six target tasks, with raw-corpus I/O, or during optimizer updates. The
optimization preserves native count/reward logic and remains demo_clean-only.

The gate executed **1,551 actions** across its six episodes. These have zero
source-training credit and are charged to the newer shared1M-action RoboTwin
campaign cap. Alongside r4's11,475actions, verified prior cost is13,026actions.
Earlier r1/r2/r3 attempts completed no policy action. Reset-internal PhysX
integration ticks were not separately instrumented; action budgets count
issued/executed robot-control commands, including any explicit settling/noops.

Initial compute accounting: r1reserve2m+r2duration14m49s+r3duration2m49s+
r4duration16m51s+this4m31s = **41m00s**, leaving at most4minutes of the
original45GPU-minute envelope. No materially larger training time was launched.
