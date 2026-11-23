# ReGround calibration partitions

At the 16:10:04 Shanghai receipt cutoff on 23 November 2026, collect01 and collect-seed0-01 had both published the same 4 policy-driven keys. The babble03 and babblecpu01 allocations had both published the same 101 motor-babble keys. Their per-key work intervals overlapped. Every pair agreed on transitions and episodes, and all 105 surviving NPZ/JSON pairs passed validation. For policy collection, the retained 23 batch checkpoints reproduced all four complete compressed shards. Seven batch paths had write receipts from both owners. A shared path can retain only its last published version. The overwritten outputs had no retained per-writer hash or copy in the inspected material. [Audit report](audit/RESULT.md) · [Production counts](audit/counts.json) · [Per-key evidence](audit/duplicate-keys.tsv) · [Final files and checkpoints](audit/survivors.json)

Two kinds of scratch paths matter here. The shared production checkpoint temporary was `batches/<key>/batchNNNN.<pid>.tmp`. Equal PIDs on different hosts could collide there. Both writers had different PIDs for each of the seven observed double writes, so their common destination was the final `.pt` file. In the installed simulator, processes sharing a literal `/tmp` can also append to `/tmp/robosuite.log` or generate `/tmp/robosuite_temp_tex/<texture-name>.png` under the same name. The evidence connects neither simulator path to a historical shard mismatch. [Production path analysis](audit/RESULT.md) · [Installed source locations and digests](tmp-paths.json)

## Parent launcher

The parent can pin this directory in a **new detached checkout**, alongside the tracked `speed/runtime.sha256`. The repository ignores new files under `outputs/`, so the parent needs to include the new files explicitly when making that pin. The [tested file digests](tested-code.sha256) identify the handoff before the parent assigns its Git pin. The launcher checks the detached wrapper revision and calls the existing detached reference runner at `23a6641562a4d833837f27ab0535a20ad94e8ec8`. These examples describe payloads inside an already allocated Baige Distributed Training container. The study's GPU budget for this handoff was zero. [Launcher](launch.sh)

```bash
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
launcher="$root/src-$TRANSPORT_SHA/$run/partition/launch.sh"
export REFERENCE_CAP_GPUH=40  # use the parent's current, possibly lower, reference ceiling

# Separate example assignments for seed 0. Bounds are zero-based, half-open.
bash "$launcher" "$TRANSPORT_SHA" low-s0-01  2 1800 collect --namespace canonical --seeds 0 --shards 0 17
bash "$launcher" "$TRANSPORT_SHA" high-s0-01 2 1800 collect --namespace canonical --seeds 0 --shards 17 25 --descending

# Other seeds, both cells, highest native babble shard first.
bash "$launcher" "$TRANSPORT_SHA" babble-hi-01 1 1800 babble --namespace canonical --seeds 1 2 3 4 --shards 35 50 --descending

# Only once seed 0 has a complete, stable pool in this namespace.
bash "$launcher" "$TRANSPORT_SHA" fit-s0-01 1 1800 fit --namespace canonical --seeds 0
```

The collection domain is indices 0–24 for policy and 0–49 for babble, in each of the two native cells and each selected seed. The high/low assignments above are disjoint **from each other**. The legacy collectors still enumerate their entire configured seed domains. Starting at opposite ends buys separation in time, and those jobs can eventually meet. For a guaranteed independent run, use a fresh namespace name in every command for that run. It maps data, batches, locks, model caches and fit receipts to `$root/partition-namespaces/NAME`. An independent namespace starts empty. Use `canonical` to resume the original data. All namespaces use the same reference budget ledger. [Selection and accounting](partition.py)

A lane processes its assigned slice without stealing. The native seed, shard number, both cell names, eight-task order, batch of four, RNG and whole-batch/whole-episode overshoot are preserved. Babble uses the native disposable camera-free child, one at a time per allocated lane. Each retry uses a **fresh attempt ID** and the same namespace and range. Already sealed pairs are validated and skipped. A busy legacy lock is revisited by the final completeness check. An outstanding key or a busy fit returns 75. Corrupt or conflicting files fail the attempt and leave the evidence available. [Collection and publication](collect.py)

The single-card fit requires exactly one seed with all 150 native NPZ shards and their JSON sidecars. Specifically, each cell needs 50 babble and 25 policy shards. The parent should establish that legacy writers for that seed have finished publishing. Fit hard-links its input snapshot, validates the native shapes, counts, holdout and mixtures, and seals the input hashes. It retains the released 100,000 nominal and 10,000 target fit budgets, 60/40 mixtures, five VLA holdout episodes per cell, four native model specifications and the saved initial RNG for each model. A fresh attempt reloads completed caches with a matching input-manifest fingerprint and resumes pending model fits. A legacy cache without that provenance requires separate reconciliation before it can be reused. A mid-model interruption restarts that model from its saved initialization. [Fit payload](fit.py)

## Publication and recovery

- New collectors also acquire the original `locks/<native-key>.lock` advisory lock. A batch writes to a unique temporary file and hard-links its complete checkpoint at the native batch path. A losing publisher replays the winning checkpoint. A process killed before linking can leave a private temporary file that future attempts ignore.
- Each completing writer first records its own NPZ and JSON SHA-256 values in `partial/<native-key>/candidate.json` inside its attempt. A validated shard pair moves from private staging into a permanent, nonempty `partition-seals/<native-key>/` by atomic directory rename on the same filesystem. Concurrent new writers elect one pair. Hard links then publish that pair's JSON and NPZ in order. Any new owner can finish a missing link after a crash. The seal and public files share inodes. Keep the seal for recovery.
- A preexisting NPZ is verified. If a legacy JSON alone conflicts with the elected sidecar, publication stops and preserves both files. Legacy code can still replace the shared batch or final paths. Schedule nonoverlapping ranges or a private namespace when that possibility affects the intended comparison. If an older writer replaces an elected public inode, the wrapper reports the conflict on verification.
- All admissions use `reference-native/attempts/`, charge open reservations at their maximum and settled reservations at their recorded GPU-seconds, and bound the new charge by cards × wall seconds. An explicit reference cap of at most 40 GPU-hours preserves the registered 8.5 GPU-hour streaming allocation and the parent's separate overhead accounting. From a fresh attempt, pin checks, reservation, probe and payload share one deadline. The exit receipt charges gross elapsed time. A preflight failure records its actual seconds with `admitted=false` in the ledger. The scheduler's billing and teardown remain part of the parent's overall ledger.
- The new launchers serialize their short reservation transactions with a nonempty directory at `attempts/.partition-reserving` and still acquire the native budget lock. Serialize admission with any **legacy** launcher at the parent. If two new admissions meet, the unsuccessful one exits before the probe; use a fresh ID after the first reservation is visible. If an owner dies inside this transaction, inspect `owner.json`, the allocation's terminal state and its reservation. Under exclusive parent control, archive the stale gate with a single rename before retry. Never clear a live gate, infer death from age/PID, or discard an open reservation's maximum charge. Ordinary shard and fit retries require no owner cleanup.

## Linux CPU qualification

The partition layouts covered all five seeds and both cells. Their exact job dictionaries matched the released CLI's 250 policy shards and the 500 inputs passed to native babble workers, with no duplicates. Across eight policy fixtures at the two end indices, the native loop ran with deterministic synthetic rollout traces. Replaying after a forced interruption reproduced the reference NPZ bytes, retained the first batch's modification time, and skipped completed pairs. In a separate same-batch race, two processes generated different payloads and both consumed the one retained checkpoint. A native process holding the original lock excluded the wrapper, then released it with the same lock inode. The fixture also exercised gross admission against an open and a settled reservation. [Qualification](qualification.json) · [Fixture](test_collect.py)

On GPFS, two CPU processes bypassed the advisory lock deliberately and raced distinct candidate payloads into one key. The public metadata and arrays came from one elected writer. Two real camera-free babble episodes at opposite endpoints also matched the directly run reference bytes and scientific metadata. The fit's five Linux fixtures used model doubles with the native data audit, cache specs and model selection. They covered incomplete pools, changed inputs, concurrent or interrupted fits and the one-card reservation. The shell fixture checked status propagation and charging for a failed pin check. [PFS publication](publication.json) · [Fit test output](fit-tests.log) · [Launcher check](launcher-qualification.json)

To reproduce the checks, copy the test files and transport Python modules to private Linux scratch, set `FIXTURE` to that directory and source `cpu-env.sh`. From there run the following commands. The publication check creates a disposable directory on Baige PFS under `/mowenjie/tmp`. No GPU runtime comparison or live cross-host synchronization experiment was part of this handoff.

```bash
py="$REGROUND_RUNTIME_ROOT/env/bin/python"
"$py" -Bu test_collect.py --output "$FIXTURE/qualification.json"
"$py" -Bu test_publish.py "$FIXTURE/publication.json"
"$py" -Bu test_fit.py -v
"$py" -Bu test_launcher.py "$FIXTURE/launcher-qualification.json"
```
