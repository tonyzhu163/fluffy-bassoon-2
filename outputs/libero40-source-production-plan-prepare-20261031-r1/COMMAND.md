# Unfunded production-plan preparation on allocated CPU

Prepared from `01a47e09e4f00881f2bc9c5b02ced59508940258`. This is metadata
preparation, not a scientific/resource approval or an additional launch gate.
No scheduler submission occurs in the launcher.

Root can deploy the exact reviewed clean commit and use an existing Baige
allocation with **0GPU/4CPU/16GiB, at most20 minutes**:

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_production_plan_cpu.sh" \
  "$SOURCE" libero40-source-production-plan-cpu-20261031-r1 \
  libero40-sac-topup-20261031-production-r1
```

The qualified Python3.12 base runtime attests zero actual CUDA devices before
masking and records CPU/memory cgroups. No tests, model construction, forward
pass, optimizer, simulator, corpus copying, funding or canonical source-ledger
write occurs. Existing `make_plan`/`parent_record` reads terminal metadata and
hashes recoveries as needed. The two exact CF49 bridge overrides are fixed;
the original48 receipt/evidence paths are resolved in the clean worktree.

Fresh output root:
`/mowenjie/fluffy-bassoon-2/outputs/libero40-source-production-plan-cpu-20261031-r1`.
It contains immutable `production.json`, its file/content SHA receipt in
`index.json`, `assignments.json` proving all40 admitted/38 assigned/two retained,
and only an **UNFUNDED** bound macro template. Cumulative source resource cap
is998,554; inherited charge457,851 is retained once. Old32h accounting is untouched.

The CPU clock reserves1,140s execution plus60s startup/teardown; interrupt/timeout
preserves partial outputs and the conservative CPU charge, with GPU charge0
only after actual attestation. Re-preparation needs a fresh output identity;
it never overwrites a partial or completed production plan. No GPU collection
starts until human25h approval and separate root funding.
