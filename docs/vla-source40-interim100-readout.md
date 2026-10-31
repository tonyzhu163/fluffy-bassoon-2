The additive `interim100-libero90` profile reuses the existing firstwave planner, launcher, candidate loader and common-motor evaluator. The default `legacy500` profile retains its98 cases /626 episodes /250k-action /6-GPU-hour declaration and fixed500 admission. No existing run or worktree is migrated.

The new profile is prepared, **unfunded and unlaunched**. It fixes all90 LIBERO90 tasks at state0 /scene seed123 /H400 +10 settling. Native and AD/DPT full/static run K0 on every task. The nine predeclared IDs0,10,…80 additionally run AD/DPT full and same-weight episode-reset at K1, with one actual support and a separate query. Static weights never replace reset controls; K0 reset reuse retains the existing identity/RNG proof.

The physical grid is522 episodes and214,020 nominal actions including settling, with220k joint hard cap. Six disjoint case-strided workers have2,700 cumulative seconds each (4.5 GPU-hours total), initial slices2,640, and action caps nominal+900:219,420 total, leaving580 joint headroom. Startup, tests, failures, retries and teardown remain inside the original worker clocks. The new profile requires an explicit graceful driver stop before its outer timeout. It adds no qualifier reset, target SGD or source actions.

Source bindings are fixed to the actual complete100-update matrix from producer `2bc15da3028ab6a3cbaed4b813f3f761a680f1d9`, run `libero40-source-topup-interim100-20261031-r1`. Matrix file SHA `dc9ebd1274170e021906c1f66623a98713405e227229bc022e03eb4c1c92a623`; CPU manifest SHA `6563d1fc76ba506cf9f44cb648c30455313a33602ab1154e30b37cc54bc54883`; assembly file SHA `89d28ad329b612ba52a4bcf108150a335de378a0e0b209e73b7bd5f47a347160`. All four policy/full-recovery hashes are checked from that exact matrix. An arbitrary100-update matrix,25-update fallback, source-generation swap or target-selected checkpoint is rejected before motor construction.

Metadata-only plan preparation, after the execution commit is reviewed/deployed:

```bash
PYTHONPATH="$PINNED/src" "$PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_target_firstwave_plan \
  --profile interim100-libero90 --worktree "$PINNED" \
  --source-commit "$SOURCE" --output "$NEW_PLAN/plan.json"
```

For actual0-GPU CPU qualification, append the existing two focused files to the CPU launcher: `--test-file tests/test_vla_source40_target_firstwave_plan.py --test-file tests/test_vla_source40_target_firstwave.py`. They cover legacy defaults, exact100 case/support counts and worker ownership, strict220k/4.5h escrow, saved100-versus500/25 rejection, fixed producer/snapshot binding, cumulative retries/graceful deadlines, and the unchanged real episode/context journal fixture. Where the shared producer artifacts exist, a metadata integration test checks their actual matrix/manifest/assembly and all eight policy/recovery file hashes without constructing a model.

Only after a separate target budget is authorized, the existing launcher argument order works unchanged:

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_target_firstwave.sh" \
  "$SOURCE" "$FRESH_RUN" "$WORKER" "$NEW_PLAN/plan.json" \
  /mowenjie/fluffy-bassoon-2/outputs/libero40-source-topup-interim100-20261031-r1/experiment/scales/update-000100/index.json \
  /mowenjie/fluffy-bassoon-2/outputs/libero40-source-pipeline-cpu-20261031-r2/assembly/assembly.json \
  "$AUTHORIZED_TARGET_LEDGER" 2640
```

The separate ledger keeps format `libero_target_wave_budget_v1` and binds this plan's `firstwave_sha256`,220000 maximum actions,16200 GPU-seconds, `source_campaign_immutable:true`, explicit `scope_amendment`, and exactly six target-only reservations. Each row binds run/source/worker/wave, its planned action cap and2700 worker seconds, `purpose:target_evaluation`, `source_pretraining_data:false` and `fresh_ppo_training_data:false`. Planning or CPU tests create no reservation.

Results use `fixed100_models`, with the exact source input bindings; legacy results retain `fixed500_models`. All raw observation/noise/action/reward/boundary records, context/global+controller+motor RNG recoveries, common-process identity and unchanged-parameter checks are inherited unchanged. Report current native /90 and memory-panel /9 separately; these are single-initialization development readouts, not full1M/500 completion, a new confirmatory test set, or a DSRL/PPO method comparison. Full Plus/Pro inventories remain separately scoped.
