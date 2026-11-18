Current steering,8November2026: the user approved100 cumulative GPU-hours and
requires an effective PPO/GRPO source control before proceeding with further
student experiments. No effective control is established. The audit and live
cost ledger supersede the earlier66h request and next-action paragraph below.
See AUDIT-EFFECTIVE-RL-20261108.md and LATEST-BUDGET.json. Budget approval alone
does not admit the old resume: its status is prepared_pending_learning_audit.
No new GPU allocation is active. Inspect actual learning signal, simulator
camera fidelity and recipe differences before choosing the next controlled run.

# Restoration handoff after the integrated first cycle

This supersedes earlier next-action paragraphs in PLAN.md without changing the
scientific contract or any closed run. Literature: docs/literature/robotwin-ppo-review.md
and docs/literature/frozen-vla-offline-icrl-review.md. Durable verified results are
in docs/FINDINGS.md; active coordination remains the single outputs/coordination.md.

Completed: a fresh eight-rank joint-Motus PPO collect/update cycle,15 Adam updates
per rank, full checkpoint1 (58 files,9,958,082,409 bytes), and DEV0/DEV1 panels3/16
and4/16. Every actual learner batch passed an independent reference. Full cold
CPU audit additionally loaded all24 worker tensors and checked CUDA RNG counts
against each actual Ray worker attestation (one assigned CUDA device per worker),
optimizer counters, case ordinal4 and retained action escrow. See
MOTUSOPTIMIZED03-CLOSEOUT.json and MOTUSOPTIMIZED03-BOUNDARY02.json.

Completed independently: all four offline students fitted1000 actor updates on
the matched source windows and passed full learner/optimizer/sampler/RNG recovery.
Five frozen controls then completed20 new real simulator episodes. All130 trace
files passed independent CPU command, physical-observation, feedback and memory
replay within original tolerances after the scoped FP32 precision fix. There was
no evaluation SGD. See STUDENTONLINE02-COMPLETE.json. Full-history imitation is
not chronological AD. These two source DEV cases establish execution, not gains.

All GPU allocations are released. Cumulative conservative creation-to-release
cost is215849GPU-s=59.9581h against the approved61h cap.3751GPU-s remain. Preserve
this full history; do not reset costs or silently launch a six-GPU-hour job.

Immediate next step: prepare/admit MOTUSRESUME01-CONTRACT.json only after the
required explicit cumulative cap increase (proposal66h) and actual CPU staging
preflight. Resume exact checkpoint1 with unchanged8-rank layout/B32 train/B2 DEV,
global2048, micro256/accum1, five epochs, same loss/optimizer schedule and case
order. The native runner loads actor/Adam/scheduler DCP, then all24 sidecars.
Actor fingerprints must match before collecting. The interrupted cycle2 physics
is discarded; copy its action journals with all resolved/unresolved charges into
the fresh run escrow. End at30 updates/rank, checkpoint2 and DEV2. New output
identity motusresume01. Launch scripts and proposal are reviewable in this folder.
Use latest reviewed main, exact Git commit, new live queue/device/storage checks,
8GPU probe, retries off and2700s creation-to-release watchdog. Keep source/model
read-only and original checkpoint/journals protected. Source validation binds all
existing model, learner, sampler, executor and renderer files. No world-size,
mask, reward, LR, precision or optimizer migration is authorized by this resume.

After a successful resume: independently read checkpoint2, new traces and
metrics; demonstrate complete fresh-data progression0→1→2. Then prepare matched
joint-Motus GRPO with the same motor/batches/seeds/engineering, and a measured
budget request if necessary. Neither a learning gain nor the full source
campaign follows from two cycles. Extend predeclared curves only after stable
runs and record actual sample/optimizer counts and critic/KL diagnostics.

Offline/scientific work still needed: genuine multi-version improvement
histories before chronological AD; qualified same-query DPT labels; support and
separate-query curves at0/1/2/5/10/20; reward/order/reset/wrong-task controls;
independent-case uncertainty; remaining source seed banks and six-source/six-held-
out transfer. Never train a held-out task to reproduce the public adjust_bottle
recipe. Disclose base joint-SFT exposure. DIT/IQL controls retain the current
single-Q/independent-encoder distinctions from their primary papers.
