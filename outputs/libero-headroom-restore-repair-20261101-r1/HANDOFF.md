# Exact DSRL recovery repair

Prepared on a fresh detached b665 worktree. Compose ONLY this package's changed/new files on reviewed main f16820e3b34e268cd22c3bbb0aaee9cc7eedf890 (or its reviewed successor). Keep main's additive worker helper. No tests, imports of JAX/Flax, model/array computation, simulator, resources or old-worktree changes ran on macOS.

The allocated diagnosis proved that the old cursor's raw signature matches the actual saved agent bytes, while Flax template restoration changes mapping order: all actor/critic/target/temperature/Adam leaves, replay, counters and RNG remain exact. The old byte guard therefore rejected a valid restoration. `restore_checked` retains checkpoint SHA, SHA(saved agent bytes)==old cursor signature, training equality, update count and replay length. It additionally requires exact canonical state dtype/shape/values, full replay identity and actor/sampler/Python/NumPy/motor RNG equality. It never samples or updates. It writes a separate proof receipt, preserving the original checkpoint/cursor bytes. Normal subsequent episode/cursor publication remains unchanged.

`onset._signature`, OfficialSAC, task57, source collector, frozen-query sampling/weight guards and the complete physical-episode/optimizer tail are unchanged. The only panel compatibility is the exact saved legacy-signature/restored-raw-signature pair after the full proof; unrelated aliases reject. The actual current301700 cursor has no active panel, and its next320k panel remains unchanged. See PROTECTION.json for four complete byte-identical modules and13 unchanged AST bodies.

The explicit migration keeps the original b665 scientific trainer, pre-SGD selection, learning contract and screen-to-b665 admission. Its runtime pin is the new qualified composite. `--execution-migration PATH --execution-migration-sha256 SHA` is required on both begin/run via the existing launcher; no arbitrary post-SGD source alias is admitted. MIGRATION-TEMPLATE.json is deliberately unadmitted until the sole owner binds final source, exact original refs/28217-second closed clock and the actual diagnosis. It only accepts this301700/15579/1720 task44 ancestry. All later attempts use the same57600-second clock, original phase identity, existing physical journals and current checkpoint/cursor. Original failed-attempt costs are not refunded. Before a first repaired reservation the current cursor must still have the original633e SHA; later same-pin continuations must extend only this same admitted clock lineage. Each new attempt records runtime/scientific pins and the migration SHA in its own phase-stage.json.

## Funded allocated CPU qualification

Use the owner's fresh180-second/0GPU8CPU64GiB guardian and actual qualified CPU runtime; export JAX_PLATFORMS=cpu before imports. No old four-inspector tests need repeating. Exactly20 new cases plus the one changed non-Flax episode-flow fixture:

```sh
python -m pytest -q tests/test_vla_headroom_restore_guard.py \
  tests/test_vla_target_online_dsrl_headroom_learning.py::test_fresh_native_readmission_aborts_before_any_sgd_and_cannot_resume_past_it
```

New cases use real Flax serialization with reordered mappings, altered actor/Adam/target/dtype/shape, counters/training/replay/RNG/signature failures, strict panel aliases and real existing-clock preparation under explicit source migration. No NN or optimizer step is constructed. The old episode-flow fixture keeps its deliberately non-Flax fake restore mocked; actual new guard behavior is covered separately.

For the ONE original431MB checkpoint guard dry-run, make a fresh SHA-bound profile with `execution_source` equal to the final composite, the exact checkpoint/cursor/contract refs from the existing diagnosis PROFILE.json and the same dsrl_root. No model/VLA path is accepted:

```sh
JAX_PLATFORMS=cpu python -m fluffy_bassoon.meta_rl.vla_headroom_restore_guard \
  --profile /ABS/GUARD-PROFILE.json --profile-sha256 PROFILE_SHA \
  --output /ABS/FRESH_CPU_OUTPUT/real-guard.json
```

It constructs only the original tiny PixelSAC on CPU, invokes the actual production guard, verifies original input hashes before/after and records zero SGD/robot actions. It requires the actual job hostname, JAXCPU and exact clean execution source. Existing outer allocation/clock/0GPU attestation is owned by finish_assembly, not duplicated here.

## Same-phase runtime continuation, only after CPU pass

Owner binds MIGRATION-TEMPLATE.json to the final exact composite and existing immutable refs; do not rewrite any old plan, admission, contract, cursor, checkpoint or worker contract. Retain original screen source and frozen-selection SHA. Existing launcher command, with only the explicit new execution binding:

```sh
bash scripts/diagnostics/vla_target_online_dsrl_headroom.sh \
  NEW_COMPOSITE dsrl-pi05-headroom-20261101-s0-r1 learn 25140 \
  --resume --code-worktree /ABS/FRESH_COMPOSITE_WORKTREE \
  --screen-source ORIGINAL_SCREEN_SHA --selection-sha256 ORIGINAL_SELECTION_SHA \
  --execution-migration /ABS/ADMITTED-MIGRATION.json \
  --execution-migration-sha256 MIGRATION_SHA
```

The launcher keeps the original worker/output identity and new unique attempt directories. Whole-phase prior GPU charge is28217/57600, leaving29383 seconds; this example reserves the usual25140+60 from that existing pool. The fixed500k endpoint/20k panels, physical cap and full actor/replay/optimizer/RNG are unchanged. This package does not launch or grant resources.
