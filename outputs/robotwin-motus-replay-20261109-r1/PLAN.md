# Teacher-forced replay of released Motus π0.5 — 2026-11-09

Question: is the native phone baseline gap (27/25 vs paper 49/53) caused by our
input pipeline, or by the checkpoint / unpublished closed-loop settings?

Feed recorded RoboTwin `place_phone_stand` expert frames (clean50 archive,
revision `3dc3b798`, episodes 0–19, every 4th frame) through the exact
baseline loader (`configuration()` from the release-baseline `evaluate.py`,
K10 ODE, checkpoint/norm hashes asserted). Compare each predicted 32-step
chunk with recorded future `joint_action/vector` targets, aligned at shift 1
(RoboTwin convention: action[k] = state[k+1]) and shift 0. Reference: a
"hold current state" chunk.

Variants share per-query noise: `base` (cv2-decoded frames as RoboTwin's own
converter, seen instruction), `rgb_swap`, `unseen` instruction, `prefix`
(Motus scene prefix), `wrist_swap` (left/right wrist exchanged, sensitivity).

Reading: base error far below hold → pipeline matches training; gap lies in
the checkpoint or closed-loop settings. Base near hold, or a variant clearly
better than base → input mismatch to fix before any RL.

Budget: one Pro6K GPU, 45-minute worker limit, within the existing Epoch2
100 GPU-hour cap (97.32 h remaining before this run). No training.
