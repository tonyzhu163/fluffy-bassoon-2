# RoboTwin re-entry: RLinf and VLRLA

2026-11-05, Asia/Shanghai. User direction: bring back RoboTwin, use RLinf and
VLRLA baselines, and anchor the work to VLRLA's six source tasks. This plan
does not restart any archived run or change an active RoboCasa contract.

## Scope

Use `handover_mic`, `pick_diverse_bottles`, `place_a2b_left`,
`place_object_stand`, `place_phone_stand`, `scan_object` for source learning.
Reserve the six unseen tasks in `configs/robotwin_vlrla.json` for transfer.
Baseline roles: common native SFT pi0.5; RLinf PPO histories; RLinf GRPO online
learning; VLRLA-bridge in-context adaptation.

Durable method: [comparison protocol](../../docs/robotwin/rl-baselines.md).
Existing audit: [VLRLA protocol](../../docs/literature/vlrla-protocol.md).

## Source audit

- Project `main` was fetched and matched `origin/main` at `ccbad585` during
  initial inspection; new execution must bind the final reviewed main commit.
  Other owners have dirty files, including coordination and findings; preserve
  those edits and stage only this work.
- RLinf upstream HEAD observed as
  `77a521298f1d215971246c2d7a423720d9e76652`. Exact API file contents and Git
  blob identities are in [UPSTREAM.json](UPSTREAM.json).
- VLRLA v1 Section 5 confirms six seen plus six unseen tasks and RLinf GRPO
  as the main-table online baseline. Appendix A.2.1 uses PPO for histories.
- Rechecked the arXiv paper and a focused code search. No official VLRLA code
  or matched checkpoint was verified. This bounded search is not proof that
  none exists. Retain reconstruction labels until artifacts are verified.
- The audited RLinf train/eval seed files cover three of the six source tasks:
  `handover_mic`, `place_a2b_left`, `place_phone_stand`, each with 1,000 TRAIN
  and 150 evaluation seeds and no TRAIN/evaluation integer overlap.
  `pick_diverse_bottles`, `place_object_stand`, `scan_object` have no entries
  in either file. [TASK-MATRIX.json](TASK-MATRIX.json) records all twelve tasks.
- The upstream loader indexes `data[self.task_name]` when a seed file exists.
  Changing only `task_name` to a missing task will fail; silent seed fallback
  is not available. Task-module availability, planner-qualified scenes and
  model competence are separate requirements.
- Current pi0.5 `adjust_bottle` PPO uses 200 controls, H50, five flow steps,
  noise .3, 256 environments x four collection rounds, five update epochs,
  global minibatch 2,048 and per-GPU microbatch 32. VLRLA's collector horizon
  is 400. Keep the original recipe and every comparison override explicit.

## Baige read-only inventory

Reused an existing development-browser tab. The project-mounted filesystem
reported about 510 GiB available at inspection. The retained
`envs/robotwin-vlrla-20261029-r1` overlay points to the older shared
`maniskill-pi05-rlinf-e1a82ebf-20261028-r1` environment. This does not establish
that the current upstream runtime, simulator assets or model weights are ready.
No storage quota/write admission, queue-capacity check, GPU allocation,
model load, simulator episode or test was performed. No other job was changed.

## Execution sequence

1. Locate and hash the retained RoboTwin source/assets and common pi0.5
   checkpoint. Audit task exposure and per-task normalization. Reuse the old
   qualified pins only with an explicit recipe/runtime comparison; use a fresh
   isolated runtime if migration is needed. Protect shared live dependencies.
2. Build explicit six-task configs from pinned RLinf recipes, retaining the
   published recipe plus a separate VLRLA H400 reconstruction. Prepare missing
   seed banks on allocated Linux using planner-qualified scenes under the
   actual embodiment/randomization; keep disjoint TRAIN/DEV/TEST identities.
   Use existing upstream banks where appropriate, recording whole-bank hashes.
   Never substitute `pick_dual_bottles` for `pick_diverse_bottles`.
3. Check storage/quota/write access, queue capacity, resource limits and actual
   GPU type. Inside the allocated job run `scripts/[train]gpu_probe.py` with
   expected count/output, then real episodes with bound renderer/inference
   devices. Verify nonblank cameras, finite H50x14 outputs, physical stepping,
   native parity and baseline weight signatures. Measure memory and throughput.
4. Measure a native source-task panel before SGD. Qualify RLinf PPO likelihood,
   value replay, frozen VLM and full recovery on actual data. Run a bounded
   source pilot, recording advancing optimizer updates and independent DEV
   success. Implement/qualify GRPO on the same pi0.5 interface and cases; an
   OpenVLA-OFT GRPO configuration alone is not a pi0.5 baseline.
5. Only after measured source improvement, collect chronological histories and
   fit the reconstructed VLRLA-bridge. Share the corpus with comparison
   students and use source-only selection. Define trial history length and
   support/query accounting before deployment; Table 1 leaves these ambiguous.
6. Evaluate the six-task source comparison, then the locked six-task transfer
   panel with full/reset and reward controls and all costs. Publish verified
   results in FINDINGS; keep raw evidence here.

Initial qualification/pilot ceiling: 4 GPU-hours including failed attempts,
within the repository's task authorization of up to 50 cumulative GPU-hours.
Bind the allocation wall limit to remaining budget before each launch. The
full six-task 25-iteration collection is not assumed to fit that ceiling;
measure capacity/update time first. Budget expansion above 50 GPU-hours needs
user approval. Monitor launched work every 20 minutes and record cluster,
job ID, advancing progress, last check, next action and cumulative charge.

## Current state

`SOURCE_AUDITED_NO_JOBS_LAUNCHED`. No baseline result or VLRLA reproduction is
claimed. Missing scene banks, common-motor verification and runtime migration
remain execution prerequisites. Keep all archived evidence immutable.
