# Center-bowl placement repair

One new simulator episode on frozen development layout0. No policy weights
are loaded, and no query layout is used to tune this teacher.

Teacher 03 established that candidate 0 grasps, lifts 14.2 cm and transports the
intended bowl, but descends while about 4 cm behind its requested wrist target.
It pushes the plate instead of reaching native `On`. The revision preserves
the actual successful grasp/lift controls and separates horizontal alignment
from descent. Both failed teacher03 continuations remain unchanged.

1. Reset the same layout0 state and seed 7; require the public initial
   observation hashes to match the frozen geometry receipt. Replay the exact
   first 133 delta controls of teacher03/layout00-0, ending at action index 132.
   Record whether the resulting state equals the saved prefix state.
2. Hold the actual controller orientation and current high wrist Z. Correct
   carried-bowl center XY toward the plate center with a maximum 5 mm world
   translation per control(0.10 m/s at 20 Hz). Start descent when bowl/plate XY
   separation and wrist high-Z error are each at most 5 mm. This phase allows
   at most 80 controls. Abort if the selected bowl loses its lift and falls
   below its initial body height plus 6 cm.
3. Continue centering with at most 2.5 mm XY correction per control, and descend
   at most 1.5 mm per control(0.03 m/s). The bowl-height target is the current
   plate body Z plus the successful native0 replay's final bowl–plate Z
   difference. Recompute wrist displacement from the observed carried-bowl
   position. Allow at most 160 controls; rotation corrections are bounded by
   0.025 rad per control throughout the servo.
4. Stop on the first intended native `On` predicate, matching BPP's native
   task endpoint, or retain the bounded controller failure. The maximum
   episode length is 373 controls. Every recorded action is the actual
   controller-scaled, bounded seven-dimensional OSC delta.

These tolerances and motion budgets define the servo and prevent descent
before physical alignment; they are not model-performance thresholds. The
teacher uses privileged simulator object poses to create data. BPP evaluation
receives the packed visual/proprio/action demonstration through its unchanged
official prompt interface.

Candidate 1 needs no new simulation: teacher03 first reaches its native goal
at action 333, so its deterministic prompt prefix contains 334 controls. The
full trajectory and its later loss of `On` remain in the evidence. Neither
prompt claims stable released placement beyond the benchmark endpoint.

Allocation: one previously qualified RPBZZZ6 GPU,8 CPU,112 GiB RAM;10-minute
scheduler cap and 420-second teacher-process timeout. The actual allocation
still runs the GPU probe. This stays within the repair stage's 4 GPU-hour cap;
before teacher04, cumulative usage is 0.0525 scheduler/0.150278 conservative
GPU-hours. Launch only an immutable reviewed `main` commit, with fresh
`teacher04` output identity.
