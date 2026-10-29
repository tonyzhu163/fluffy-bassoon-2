# Corrected saved-trajectory PPO diagnostic

Same separate batch3/K4/five-epoch objective, immutable r4 input and unchanged
likelihood tolerances as r1. No simulator/VLA/new robot actions. Restore the
exact process-global float32 matmul precision high set by the original pinned
motor; record both default and active precision plus replay errors before the
gate. Require finite replay, changed weights, nonzero finite gradients, one
completed PPO update/five optimizer steps, and a full diagnostic recovery.

Originalr1 remains intact and failed before SGD. A verified scheduler receipt
replaces its earlier startup allowance120s with actual49s: total prior actual
usage2590seconds,110seconds remain of2700. This retry has oneGPU, one scheduler
minute,45s process alarm,50s timeout/5s kill grace. Source checkpoints and
batch8 matched comparison remain unchanged. If this attempt cannot finish,
preserve the full original recovery and stop; do not fund another allocation
from an assumed larger budget. See COMPUTE-ACCOUNTING.json.
