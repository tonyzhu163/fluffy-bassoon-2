# Native calibration qualification passed; history-learning comparison running

The two OpenDrawer development scenes completed all 18 native episodes and
7,477 delivered controls. Both exact action-tape replays and fresh closed-loop
repeats reproduced their nominal success, length and terminal physics hash.
All perturbed branches matched nominal physics and policy observations at the
fixed correction decision.

| Scene | Hidden offset | Correction −4 cm | No correction | Correction +4 cm |
| --- | --- | --- | --- | --- |
| 982101 | −4 cm | Failure | Failure | Success |
| 982101 | +4 cm | Success | Failure | Failure |
| 982103 | −4 cm | Failure | Failure | Success |
| 982103 | +4 cm | Success | Failure | Failure |

This establishes that opposite attainable corrections are useful under the
two hidden calibration conditions, with identical current decision inputs.
It does **not yet demonstrate learned history use**: the qualification executes
all corrections explicitly. The current run next tests a reference selector
and frozen history readers trained on eight admitted scenes and evaluated on
eight other admitted scenes, with three fit seeds and matched interventions.

This is a custom calibration perturbation on scenes selected for unperturbed
motor success. Correction timing uses simulator handle geometry, and the
learned selector chooses only a correction direction at that fixed decision.
Results will not be an ordinary RoboCasa benchmark score or evidence that the
previous large VLA adapters now adapt successfully.

The gate used source `38eaf99b`, exited zero at November 17, 11:26:35 Shanghai,
and consumed 0.284 conservative GPU-hours including startup. The continuation
at `02d6071e` is running `panel01` on the same allocated RTX 4090. At 11:29,
the first screening episode completed and was retained as an unperturbed
failure; scene 982101 is next. Total phase use is about 0.324 of 8 GPU-hours.
The original campaign remains closed.

The continuation proceeds to the small fits only if the training-split
reference benefits from feedback over both the best fixed action and the
current-observation reference. Held-out outcomes do not select fit settings.
A thread heartbeat checks every twenty minutes until completion, failure
diagnosis or budget expiry. Native collection and fitting stop under the
bounded continuation; no further PPO teachers are being trained.

[Native records](evidence/gate01/SUMMARY.json) ·
[Qualification readout](evidence/GATE-READOUT.json) ·
[GPU probe](evidence/gate01/gpu.json) ·
[Exit and elapsed time](evidence/gate01/clock.json) ·
[Constructed CPU numerical checks](evidence/CPU-QUALIFICATION.json) ·
[Continuation launch](evidence/CONTINUATION-LAUNCH.json) ·
[Execution contract and limitations](EXECUTION.md).
