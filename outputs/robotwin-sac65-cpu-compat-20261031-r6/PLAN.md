# Diagnosed shared-runtime path correction

**Prepared; not allocated or qualified.** R5 reached its tests phase, then
failed at `import torch` before pytest. Its actual job `job-ykpoq2b043kr`
charged 71 scheduler seconds. This candidate fixes that observed import path
and prepares one residual attempt within the same R5 CPU grant. The execution
owner retains final resource admission, deployment and launch.

## Exact cause and correction

The existing venv has `include-system-site-packages=false`. Torch is supplied
by one literal `shared_runtime.pth` path into an existing shared environment.
The isolated `-I -B -S` worker deliberately skips `.pth` execution and previously
added only the venv site directory. Therefore Torch was absent from `sys.path`.
R5 performed zero fixture cases, projection, export, robot actions, VLA queries,
SGD updates or fits; its owned teardown was clean.

Keep `-I -B -S`. After the import firewall and hash-bound actual zero-GPU receipt,
read the single approved path as data and explicitly add its existing directory:

```text
/mowenjie/fluffy-bassoon/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/lib/python3.11/site-packages
```

The worker requires the exact metadata bytes in
[shared-runtime-path-metadata.json](shared-runtime-path-metadata.json), SHA
`715a0ab60975cfa63a3d19ce4bcd5cd58c355db799aa1b6884f4027765c49322`.
It verifies the existing venv's `shared_runtime.pth` SHA
`a90c8566fa30a6f7d0573409a586c68c16e39146327a8e7b588ae482fc8eaade`,
its single absolute literal line, the exact `torch-2.11.0+cu128.dist-info` name,
and Torch's `__init__.py` SHA
`0387d8b811b289287479c8bfdf4e1dac3a71b246f938d82da1331cf2dc8bf001`.
Tests/export additionally verify the loaded Torch origin against that file.
Phase receipts record these proofs and the actual Torch version/origin.

No `.pth` file, `_virtualenv`, distutils hook or `sitecustomize` is executed by
path setup. No generic base/system-site fallback is added. No package is
installed and no environment is moved or modified. The existing simulator/VLA
firewall, CUDA initialization ban and Adam/AdamW step bans remain unchanged.
The metadata is from file inspection, not a successful runtime import test.

## Same grant, explicit residual

The old R1–R4 envelope remains **CLOSED: 620 actual + 1,180 unused sealed =
1,800 seconds**. All four accounting snapshots are byte-identical.

The separate R5 grant now has **71 actual + 1,680 proposed R6 cap + 49 unused
sealed = 1,800 seconds**. Its actual receipt is copied without alteration to
[predecessor-r5-accounting.json](predecessor-r5-accounting.json), SHA
`9be36ce2ee0e0469d60723a83b481db43505ef8c7f35fbf7c032f3c62233060f`,
from evidence commit `553d0fd1048a2d0edd196ba42790ca49ee2d7814`.
The [budget](budget.json) and loader enforce these two separate ledgers.
There is no fresh 1,800-second grant, reset, refund, old-seal borrowing or
automatic resubmission. Combined maximum consumption is 2,371 scheduler seconds
/18,968 CPU core-seconds; actual R6 cost remains unknown.

R6 requests 0 GPU /8 CPU /64 GiB /shm2 /28 minutes, autofault OFF. All setup,
bootstrap, scientific phases and owned cleanup share the original 1,580-second
parent deadline. Bootstrap120, local-code staging30 and cleanup30 stay inside
it. One outer timer uses TERM1,650/KILL1,660 below scheduler1,680. No timer resets.

Scientific floors stay 960 before tests, 750 before projection, 660 before
export and 45 before audit. Setup can use at most 620 seconds before tests.
R5's observed 36.489 seconds to tests would leave 1,543.511 shared seconds,
583.511 above the pretest floor. This is admission headroom, not measured full
bank throughput or a promised finish. Tests180/projection900/export900/audit90
ceilings and all later-phase reserves remain unchanged.

## Frozen qualification scope

The same 150 selected fixtures remain unexecuted. No tests are added, changed
or skipped. Complete source65 /26 roots /390 K4 /1,560 episodes /651,638 actions,
source checkpoint and ancestry hashes, native masks, teacher-label/token math,
paired AD/DPT historical fields, full-byte projection and final audit are
unchanged. Teacher competence and source100 remain unmeasured/unadmitted.

Keep the exact local Git staging, full clean-status check, inherited Git
override filtering, durable original PFS input references, owned process guards,
visibility probe and complete-detail CUA helper. R4's observed inventory was
143,628 files /56,812,446,592 bytes; its conservative 8x read estimate was
454,499,572,736 bytes. These are diagnostic receipts; this run repeats full
checks with a fresh allocation proof and never reuses ended temporary code.

Only static syntax, byte/AST and hash checks are performed on macOS. Actual
fixtures and full-bank qualification run on the owner's allocated Linux job.
All 150 fixtures, full source SHA/coverage, paired export/audit, clean owned
teardown and reconciled scheduler cost are required for qualification.

Follow [CUA-HANDOFF.md](CUA-HANDOFF.md) after one narrow exact-path/accounting
review and owner admission. No new pilot, generic hardening or scientific
expansion is part of this correction.
