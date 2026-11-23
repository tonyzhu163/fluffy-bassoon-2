# ReGround 90° paired-opening sensitivity

**Baige CPU qualification passed at `84b4d980eb85dafae718fe4895ef51d0f1648140`.** For each of Base, frozen SymMeta and native ReGround CEM, the fixture compared one complete 16-assignment shard against the unmodified evaluator. All five trace arrays and all non-timing CSV fields agreed byte for byte or as serialized. Capture made four query-zero predictions. Every scored fixture batch independently reset, consumed its verified full opening and called its policy again at controls 10 and 20. The fixture rejected changes to each camera, EEF, jaw, simulator snapshot, task, episode nonce and noise tensor. It used the archived reader and private synthetic simulator, policy and cache weights. GPU runtime, historical camera agreement and scientific outcomes still require the parent's allocation. [CPU receipt](cpu-evidence.json) · [Fixture](fixture.py)

## Which assignments to run

The original Base and streaming evaluations each finished 90 shards and 1,400 assignments. Exactly 1,393 ten-control prefixes matched. The original robot reset and episode nonce agreed on all seven divergent assignments. A read-only check against the raw shards recovered these two targets, all with rep 0.

| Adaptation seed / native shard | First-proposal mismatch | Entire shard to evaluate per arm, in order |
| --- | --- | --- |
| 0 / 004 | `10_3` starts 44–49 | `10_3` 44–49, then `goal_1` 15–24 |
| 1 / 015 | `spatial_0` start 48 | `spatial_0` 45–49, then `spatial_2` 15–25 |

When all fitted original outputs arrive, run the registered three-way pairing on the **original** stores. If it fails, retain its failure and use the original-grid audit to report discordant keys and pairwise bounds. For a binary mean contrast, if the concordant cases contribute signed count `D` and `m` assignments remain unresolved, the all-grid interval is `[(D−m)/1400, (D+m)/1400]`. For the seven current Base/streaming cases this gives a half-width of 0.5 percentage points. Select the union of whole shards containing any original first-proposal disagreement across arm pairs, independently of outcomes. The runner rechecks this condition for each requested seed/shard, including newly implicated fitted shards, using the registered native ordering and batch boundaries. Report sensitivity results separately with their own denominator. [Registered pairing](../symmeta_adapter.py) · [Full-grid audit and bounds](../native_grid.py)

## Parent launch

Deploy the tested SHA into a fresh detached Git checkout containing this directory and `../speed/runtime.sha256`. The scientific modules are imported from the existing detached `cab855a0` checkout. The tested Git bundle, private detached checkout, log and preserved fixture traces are on `baige-dev` under `/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1/common-opening/fixtures/cpu-X5DeO61f/`. The bundle can be fetched into a new checkout that already has its prerequisite history.

Use one exclusive RTX 4090 on one Baige Distributed Training master. Set a 20–45 minute scheduler limit and a matching 1,200–2,700 second launcher bound. Each invocation below is a **separate allocation** with a fresh attempt ID. The deadline includes the exclusive training probe, CUDA/PCI/UUID verification, EGL binding, strict policy load and evaluation. Charge these attempts to the existing reference ledger, capped at 40 GPU-hours, within the parent's cumulative budget. Serialize admission with any legacy launchers. [Launcher](launch.sh)

```bash
pin=84b4d980eb85dafae718fe4895ef51d0f1648140
run=outputs/reground-rotation-reference-20261123-r1
launcher="$checkout/$run/common-opening/launch.sh"
export REFERENCE_CAP_GPUH=40
bash "$launcher" "$pin" paired-a-s0-01 2700 part-a 0 4
bash "$launcher" "$pin" paired-a-s1-01 2700 part-a 1 15

# After each seed's fit is published, on the same code pin.
bash "$launcher" "$pin" paired-rg-s0-01 2700 reground 0 4
bash "$launcher" "$pin" paired-rg-s1-01 2700 reground 1 15
```

Part A runs `capture`, `base`, then `symmeta` with one VLA load and a fresh native simulator pool per stage. Each stage is also an individual launcher phase. After timeout, inspect the attempt and retry the first unpublished whole stage under a fresh ID. Fitted evaluation constructs the released controller once per shard, retains its CEM generator across the native four-lane batches, loads all four receipted cache files with fitting disabled and records their hashes. Its own controller issues every action from control zero. [Runner](runner.py)

Completed stages publish atomically under `$REGROUND_RUNTIME_ROOT/common-opening/shards/seedN-shardNNN/`. Partial work and mismatch evidence stay under `common-opening/attempts/ID/`. The only accounting writes outside that namespace are the shared budget transaction and fresh reference ledger receipt. Capture saves both raw cameras, every observation leaf, task text, episode and chunk seeds, the actual native noise tensor, settled robot and full simulator state, and all 50 × 7 proposed values per lane. Before consuming a tape each arm verifies the live reset, policy inputs and tape digest. The original traces corroborate the initial 25-vector and nonce, and streaming calibration supplies the historical agent view, EEF and jaw. The original Base archive does not supply both camera frames or complete simulator state. A mismatch stops the stage and identifies the field and batch for diagnosis. Any separately justified reset/observation control needs its own provenance and the same cases. [Opening interception](opening.py)
