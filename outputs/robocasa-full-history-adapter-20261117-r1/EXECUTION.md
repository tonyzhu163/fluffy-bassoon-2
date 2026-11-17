# Execution

November 17: user explicitly authorized **up to 200 GPU-hours** and execution.
This is the total new campaign cap, including failed jobs and all follow-ons;
earlier closed campaigns remain separate. Initial qualification reservation:
four independent Pro6K GPUs, at most one hour each. No automatic retries.

Main reviewed at `dae30718`. Native qualification code is being prepared and
will be commit-pinned after CPU checks. The live queue showed 49/416 free GPUs;
actual capacity and personal quota must be checked again at submission.
Project PFS had 87 GiB available; cold archive mount was writable. No large
downloads or old-data deletions are needed. Job-local runtime probes and native
episodes remain required.

Implementation decisions:

- Keep calibration in the environment, hidden from student inputs. Update its
  onset/end at the five-control decision boundary, explicitly migrating from
  the pilot's per-control grasp stop. This permits same-query teacher labels
  for the five commands before replanning. Qualify the changed timing.
- Record visual observations and frozen features at decisions. Keep the five
  executed commands separate from the 50 proposed commands; history padding
  is constant, and the loss uses the delivered-coordinate mask.
- Qualification checks the frozen Torch motor on old development scenes and
  uncorrected/full-teacher pairs over lateral, depth and diagonal conditions.
  No new held-out learner result enters this engineering stage.
- The core loss-mask regression test protects the numerical mean over valid
  controls and its equivalence under execution-batch accumulation.

November 17, 17:51 Shanghai: native qualification submitted as
**`job-o71yh82weghm`**, `mwj_1117_full_history_qual4_r1`, four independent
one-GPU workers, 16 CPU cores and 128 GiB RAM each, 60-minute scheduler limit,
automatic retries off. Queue at submission: 51/416 free GPUs, personal quota
32/32 available. Source `5d919c390adfd231703c42e353017c62246cdb33`, deployed
through a Git bundle into an immutable detached checkout. The runtime is the
previously qualified RLinf `8f9bfa7661928ef8c17494b04f4d7bf3125634e1`.

Linux CPU validation: **30 tests passed in 85.81 seconds** in
`tests/test_flow_bridge_train.py` and `tests/test_flow_history.py`, CUDA hidden.
All four in-allocation GPU probes passed. Workers are loading the native motor;
completed simulator episodes remain pending. Output identity: `qual01`.

Conservative charge/reservation: **4/200 GPU-hours**, including startup and
barrier time. This is the complete maximum for this job, not measured elapsed
compute. Next check: 18:11 Shanghai or sooner on a startup failure. Continue
with teacher/control results, then the privileged-input adapter qualification.
