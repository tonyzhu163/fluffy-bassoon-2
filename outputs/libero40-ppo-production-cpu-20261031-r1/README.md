# Dedicated CPU tests of PPO production continuation

**Prepared, not run here.** No tests or models were executed on the Mac or
Baige development host. Root will deploy this exact reviewed commit to a new
Baige allocation with **0GPU,4CPU,16GiB, at most20minutes**. This is CPU validation
of the new continuation glue, not another GPU qualification or production run.

From the exact clean deployed worktree:

```bash
bash outputs/libero40-ppo-production-cpu-20261031-r1/run_cpu.sh \
  EXACT_COMMIT FRESH_CPU_RUN
```

The dedicated launcher uses the qualified Python3.12/cp312 base runtime at
`/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python` and the existing
pytest overlay. It reuses `require_zero_gpu` and `cgroup_resources`: actual
CUDA-visible device count must be0 before applying a CPU-only mask, CUDA must be
uninitialized, and measured cgroup limits must match4CPU/16GiB. It does not append
fixture tests to the generic metadata-only helper's inaccurate
`model_construction=false` contract.

The honest dedicated receipt declares that **tiny CPU `nn.Linear` and Adam test
fixtures may be constructed**. There is no VLA/policy construction, policy forward,
backward, optimizer step/SGD, simulator, source rollout or GPU use. Only
`tests/test_vla_source40_batch14_production.py` and the new actual shared-checkpoint
integration test are executed; existing warmstart/batch14/JAX qualifications remain
separate evidence.

The shared-PFS test binds the actual qualified FULL producer
`376415e6915399436cb3fea214c50bd3e41f32d5`, recovery
`c689a2ddb190510ba09452ad9c755446bf605351d22b41017c75187834e90ee0`,
and its actual hash-bound history and native receipts. Within `pytest-tmp/` it
creates an explicitly proposed production plan and exercises the real migration
function, copying physical journals and changing only declared phase metadata.
It verifies every non-contract leaf by exact recursive fingerprint, including
cursor14, native update1, Adam478, complete pending state and typed motor RNG,
and checks the copied9,272-action prefix and unchanged original bytes. An exact
repeat must leave the migrated checkpoint bytes unchanged. `RL2Config` construction
is a dataclass; no VLA, policy or optimizer object is created by this shared-data
test. Temporary checkpoint copies remain test artifacts, not production candidates.

In an ordinary checkout without Baige PFS, that integration test would explicitly
skip. **It has not been run or skipped locally in this handoff.** This launcher
sets `FLUFFY_REQUIRE_PPO_SHARED_CHECKPOINT=1` and requires zero JUnit skips, so
missing or changed actual shared evidence fails the allocated job. A successful
run writes `shared-checkpoint-integration.json` with the real producer/recovery,
plan/proof/checkpoint hashes and exact leaf/cursor/Adam checks.

The dedicated clock includes setup/tests/I/O/teardown plus60seconds startup
allowance, charges0GPU-seconds only after actual zero-GPU attestation, and records
any overrun honestly. Tests have a1060second phase deadline, outer execution is
1100seconds plus20second kill grace, leaving teardown margin inside the1200second
CPU-worker ceiling. Original artifacts are read-only. New logs, JUnit, cgroup/GPU
attestation, clock, integration proof and temporary copied recoveries stay under
the fresh CPU run output; failed identities are preserved rather than overwritten.
