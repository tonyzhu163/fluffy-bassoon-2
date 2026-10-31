# Phone demonstration stage0 — CPU-only preparation

No capacity GPU admission is implied. This script imports no model/simulator,
opens no action ledger and creates no robot actions. The source parent remains
sole allocator/observer. It may run once on actual allocated Linux with0GPU
and <=300CPU-worker-seconds inside the existing phase residual; the script's
own shared240-second deadline includes imports, file/source inspection,
download/copy, extraction and final inventory. Preserve failed partial output;
no timer reset, automatic retry or alternative dataset.

Entry: `scripts/diagnostics/robotwin_capacity_data_stage0.py`.
It verifies the exact public phone clean50 archive255,542,392bytes/SHA
745a98306a28fd5e112388489b1973c43b9bb9946233c7ab4f1e40c6d1b5d065,
revision3dc3b798668feb99ac61cc9086d84cbcc3d79186. A verified existing input
cache may be supplied; it is copied into the phase output and, when outside
the phase, its original bytes also count against the64GiB ceiling. Otherwise
the script downloads directly from the canonical public URL without auth.

The zero-GPU ticket must bind hostname, gpu_count0, source_commit, fresh output,
maximum_cpu_worker_seconds1..300, phase_id=frozen-pi05-endtoend-20261031-r1.
Actual scheduler/resource attestation and outer timer remain the sole parent's
existing launcher responsibility; the script does not manufacture them.

```sh
python scripts/diagnostics/robotwin_capacity_data_stage0.py \
  --source-commit EXACT_PIN --allocation ACTUAL_CPU_TICKET.json \
  --allocation-sha256 TICKET_SHA --max-seconds 240 \
  --phase-root /mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1 \
  --output /mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/capacity-data-r1 \
  --robotwin-root /mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin \
  --archive-cache EXACT_VERIFIED_PHONE_ZIP \
  --a1-corpus EXACT_IMMUTABLE_R1_CORPUS --a1-corpus EXACT_IMMUTABLE_R2_CORPUS
```

Output retains safe ZIP member/CRC inventory, canonical numeric episode indices,
whole HDF5 files with hashes, actual datasets/attributes/shapes/dtypes, numeric
14D first/last/ranges, timestamp/seed/instruction metadata and pinned author
writer/collector source excerpts. Exact explicit success metadata selects the
first8train/next4validation whole successes. If the archive omits that flag,
the first12candidate files remain available and success/temporal alignment
stay PENDING author review rather than being invented. No HDF5 current-state
field is silently relabeled as a future action. The parent has independently
identified `joint_action/vector` as drive-target state from pinned robot.py;
the exact author temporal conversion and actual archive assets still need
to be bound after this inventory. Stage0 exit0 is inventory success, not
transform/motor/teacher/student qualification.

Optional gripper statistics read only SHA-bound complete A1 decision/native
NPZs, select actual native valid-mask rows and summarize pre-actuator6/13
ranges/quantiles/out-of-[0,1] counts. No corpus frame forest, VLA or live
CampaignBudget constructor is touched. Data extraction is bounded to512MiB
per whole demo and8GiB inspected uncompressed data inside the original64GiB
aggregate, with no truncation of an admitted episode.

Prepared tests (actual allocated CPU only):
`python -m pytest -q tests/test_robotwin_capacity_data_stage0.py --junitxml=stage0.xml`.
Only AST/whitespace checks have run on macOS. Full reactive trainer code remains
a separate unqualified candidate and must not be launched from this stage0 pin.
