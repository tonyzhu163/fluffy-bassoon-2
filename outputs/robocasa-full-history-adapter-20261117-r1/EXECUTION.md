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


November 17, 18:11 Shanghai update:

| Attempt | Source / job | Outcome | Conservative GPUh |
| --- | --- | --- | ---: |
| qual01 | `5d919c39`, `job-o71yh82weghm` | Four workers stopped before controls: compact Gym state16 was incorrectly passed to the native state25 port. Scheduler 17:49:58–17:53:41. | 0.25 |
| qual02 | `8db32972`, `job-6erpto17om93` | Single worker executed its first chunk, then correctly rejected state/image sample drift caused by forced observable refresh. Scheduler 18:02:01–18:04:12. | 0.04 |
| qual03 | `feeaf3a5`, `job-elqtm2aq1oe9` | Running since 18:06:42, one Pro6K GPU. Cached observations now match. Nominal success and first 2 cm offset teacher/control pair completed. Eleven-episode rank-0 shard continues. | 1.00 reserved |

Both failures are diagnosed, retained and terminal; their worktrees are
unchanged. Native preprocessing tests also passed: 5 tests in 266.33 seconds
on allocated Linux CPU, in addition to the 30 bridge checks. Current
conservative completed-plus-reserved total: **1.29/200 GPU-hours**.
The next launch covers only ranks 1–3 with the same qualified collector,
reserving at most 3 additional GPU-hours. Initial data is 12–18 MB per episode.

November 17, 18:23 Shanghai: remaining qualification submitted at 18:16:28 as
`job-k72bbb17v39j` (`qual04`), 3 one-GPU workers, source `680b3dd6`,
60-minute scheduler limit, retries off. Queue 45/416 free; personal 27/32.
All three in-allocation GPU probes passed. The rank-0 `qual03` shard completed
all eleven episodes in 0.178 worker GPUh; provisionally charge 0.25 until
scheduler closeout. The 3/4/6 cm negative-depth teacher succeeded where the
uncorrected policy failed; these are development outcomes only.

Rank 2's nominal scene 982107 failed, so its qualification ended before any
perturbed pair. Replace this development-only scene with already nominally
qualified 983001 in fresh `qual05`, preserving the failed nominal result.
This selects on nominal eligibility, never perturbed outcomes. Reserve at
most one further GPU-hour. The privileged-input fit uses only the eight
training conditions from qual03/04/05, never held-out magnitudes/directions;
reserve at most two hours for its initial 1,024-update fit. All remain within
the 200-hour campaign cap (maximum completed plus pending reservations 6.54).

The adapter uses the planned six-layer width-256 rank-4 all-prefix-layer KV
interface. Its privileged positive control receives the active offset;
real history/current-only students have no such input. Native zero-output
parity and decoded teacher-command error are measured before/through fitting.
The sequence encoder now uses the core's equivalent parallel causal path;
a numerical regression checks values and gradients against incremental
inference, including episode resets and memory-window truncation. The run
sampler retains complete support episodes, their terminal feedback, and only
the causal query prefix, with balanced scene/condition query coverage.

November 17, 18:30 Shanghai: source `7b2da26f` passed all
33 bridge/history Linux CPU tests in 83.08 seconds. The per-run causal sampler
check also passed: complete support/terminal feedback, no future query or
attempt leakage, matched current-only clearing, balanced query coverage and
exact sampler continuation. Rank 1 and 3 completed their eleven episodes;
qual04 allocation-end reports total 0.581 worker GPUh including rank-2 barrier
wait. Provisional conservative charge 0.70 GPUh pending scheduler timestamps.

Replacement submission `job-zxra5yrwo4c3` at 18:25:32 failed at 18:26:18 before
creating an output directory. Scheduler-stored command had a new command
prepended to the old one: native editor AX setValue looked correct but did not
update Monaco's submitted model. Charge 0.02 GPUh. Use fresh `qual06`; verify
actual keyboard replacement and the submitted command, not only AX value.
Current provisional completed charge: 1.26 GPUh (0.25+0.04+0.25+0.70+0.02).
The next one-hour qualification plus two-hour privileged fit reserve 3 more.

Fresh train/development/test candidate lists are frozen in SPLITS.json;
nominal-only eligibility and balanced condition assignment are predeclared.
No confirmatory test outcomes have been opened. The privileged 1024-update fit
also runs fresh attempt-2 native SFT/teacher/adapter episodes if time remains.

November 17, 18:41 Shanghai: `qual06` is running as `job-43l5z513d967`,
created 18:32:13, one GPU, source `6c460e90`, one-hour scheduler limit.
The actual run directory and passed probe verify the corrected command.
The replacement nominal and both negative-lateral training pairs completed;
the teacher succeeds in both while uncorrected SFT fails. All sixteen
training-condition episodes (eight offsets, two arms) are now available.

`priv01` launched at 18:38:16 as `job-s41w9yvu9458`, one Pro6K GPU,
120-minute scheduler limit / 115-minute worker limit, retries off,
source `6c460e90`. Queue showed 9/416 free, personal 11/32; PFS had
84.23 GiB available. In-job probe passed. The initial adapter exactly matches
the frozen motor on all 32 decoded diagnostic queries. Update 32/1024 reached
with finite decoder and adapter gradients, 0.55 seconds/update and 7.46 GB peak
CUDA allocation. Initial active XYZ teacher MSE is 0.12922; inactive error is
zero. Later command diagnostics and fresh native attempt-2 comparisons are
pending. This is an explicitly privileged interface check, not a history result.

November 17, 18:50 Shanghai: all 20 development offset pairs
completed. Teacher success 20/20; uncorrected success 4/20. The condition set
remains exactly the predeclared 8 ID + 12 held-out-value conditions. All signs,
axes and diagonal directions remain, including the clipped 6 cm cases.
These use three development scenes and are engineering evidence only;
QUALIFICATION.json preserves the case-level results and failed nominal screen.
qual06 worker allocation ended with 0.173 GPUh; conservatively charge 0.25.
Completed provisional campaign charge is now 1.51 GPUh, plus the 2-hour
priv01 reservation. At update512 the privileged adapter's active decoded XYZ
MSE is 0.12884 versus native 0.12922: command movement remains very small.

Next engineering check: at most 0.25 GPUh to measure decoded command authority
at fixed adapter scales on training-condition development queries. This does
not open held-out learner outcomes or modify the running fit. Refit only from
that interface evidence. Main collection remains contingent on useful native
control. A nine-constant development correction baseline is implemented but
not launched; it has privileged window timing, never the hidden condition.
The main public-history fits will use disjoint development queries for curves,
and the larger execution batch of8 will be measured against the saved logical
batch8 (priv01 peaked at7.46GB on the97.9GB allocation). Core/model architecture
and optimization budget are unchanged.
