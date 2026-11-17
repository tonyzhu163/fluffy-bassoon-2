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

Status: preparing implementation and Linux CPU validation. GPU use: **0**.
