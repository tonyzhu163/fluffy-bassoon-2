# Native ReGround rotation runner

Entry points are [reference.sh](reference.sh) for Baige Distributed Training, [reference.py](reference.py) for the native phases and read-only status, and [reference_cpu.py](reference_cpu.py) for CPU qualification. The source pin is `16a98486f560676b9036d1a6d067b9dd51a5917a`. The pinned checkpoint is `lerobot/pi05_libero_finetuned@7152f13e34c6c115d808aea0e5d08aca74eac14b`. The authors’ historical checkpoint revision is unknown. The shared runtime owner supplies the verified offline cache and the environment. The parent owns scheduler submission, monitoring and the combined budget. [Protocol and published targets](REFERENCE-PROTOCOL.md) · [Runtime qualification](RUNTIME.md)

## Scheduler command

The tested code is deployed in the detached checkout below. Use one Baige master with **2, 4 or 5 Pro6K GPUs** and no additional nodes. The parent also registered the RTX 4090 queue. The probe records which accelerator produced each attempt. The executable takes the checkout SHA, a fresh attempt name, allocated GPUs, wall seconds, phase and optional seeds. Omit seeds to run 0–4. Each lane uses one CUDA device and the matching NVIDIA EGL device. The isolated job container provides device visibility. Set the scheduler wall limit as well as the runner limit.

```bash
root=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
pin=23a6641562a4d833837f27ab0535a20ad94e8ec8
run=outputs/reground-rotation-reference-20261123-r1
runner="$root/src-$pin/$run"
export REGROUND_RUNTIME_ROOT="$root" REFERENCE_CAP_GPUH=40

# Example bounded collection allocation, 2 GPUs for up to 30 minutes in total.
bash "$runner/reference.sh" "$pin" collect01 2 1800 collect
```

For the full pipeline use `all`. The phase order is `babble collect fit base reground`. Each phase can also run separately. For example, `fit01 5 3600 fit` fits all five seeds once their data are present. `base01 4 1800 base 0` runs the official frozen base for seed 0, without fit dependencies. Supply a fresh attempt name on each submission. A resumed phase can use a different supported GPU count. Keep active checkouts and attempts immutable.

The example reference ceiling of 40 GPU-hours matches the parent's initial reservation. It includes prior reference attempts. The parent accounts for the other study and scheduler overhead against the combined 50 GPU-hour allowance. The maximum charge for a proposed payload is GPUs × wall seconds / 3,600. The shared reservation lock also charges earlier unfinished attempts at their full reservation. The wrapper measures actual elapsed payload time including probes and idle lanes. Scheduler billing can include additional container startup and teardown. The full study has no demonstrated completion time until policy collection, fitting and ReGround rollouts have run.

## Work and outputs

| Phase | Released work for all five seeds | Published evidence under `$root/reference-native/` |
| --- | --- | --- |
| Babble | 2 bodies × 5 seeds × 50 shards of at least 2,000 transitions | `runs/data/<cell>/seed<N>/babble/shard*.{npz,json}` |
| Collect | 2 bodies × 5 seeds × 25 shards of at least 4,000 frozen π₀.₅ transitions | `runs/data/<cell>/seed<N>/vla_pi05/shard*.{npz,json}` |
| Fit | Nominal and deployed forward ensembles and IDMs, 4 models per seed | `runs/models/<hash>/{model.pt,spec.json,fit_metrics.json}` and `fitted/seed<N>.json` |
| Base and ReGround | Each has 5 seeds × 8 tasks × 35 starts, or 1,400 outcomes | `shards/seed<N>/rot_z/90/<method>/shardNNN/{results.csv,manifest.json,traces/}` |
| Summarize | Exact complete grid, both methods | `runs/eval/pi05/rot_z/90/<method>/<cfg-id>/seed<N>/test/results.csv` and `summary.json` |

The README requests one million transitions from each collection mechanism across both bodies and all five seeds. Babbling keeps its final whole episode. VLA collection keeps its final four-episode batch. The runner measures both overshoots, checks the original task sequence, shard count, transition count, finite values and `X[T+1,25]`/`A[T,7]` shapes, and runs the released round-robin merge. The last five VLA episodes per body/seed are held out. Each nominal fit consumes the first 60,000 babble and 40,000 eligible VLA transitions. Each target fit consumes 6,000 and 4,000. Forward and inverse models reuse those inputs. The exact 90° recipe uses the IDM anchor and disables the optional gate. [Released README and configuration](https://github.com/umersiddique94/ReGround/tree/16a98486f560676b9036d1a6d067b9dd51a5917a)

The test grid uses the original 8 tasks, starts 15–49 and replicate 0. Native horizons are 280 for spatial/object, 300 for goal and 520 for the long suite. Each method receives the released per-start noise, two native camera views and a new frozen-policy chunk every 10 controls. Native success comes from LIBERO’s task predicate. Unsafe is the union of non-target displacement, falling, contact and any speed violation above the released π₀.₅ envelope of 0.27572885155677795. Table contact is recorded separately. The final check recomputes speed violations and unsafe from stored evidence and verifies every key and trace length before aggregation. The upstream `summarise` computes success/unsafe IQM over 40 task-by-seed means and 95% task-stratified hierarchical bootstrap intervals with its 10,000-draw defaults. The output also contains per-task rates for comparison with the published table. [Native rollout](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/eval/rollout.py) · [Released statistics](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/eval/stats.py)

## Transport and resumption

- The release’s `launch()` reuses list, status and log filenames for different seeds/methods. Its claims can also survive a move to another host. This runner calls the released worker functions under per-unit filesystem locks and uses private attempt directories. A finished NPZ or shard directory is published by atomic rename. Original simulator, policy, method, fit and statistic sources stay at the upstream pin.
- The native launcher sorts and slices each 280-episode evaluation into seventeen 16-episode shards and a final 8-episode shard. ReGround’s seeded CEM generator restarts for each shard and advances across its four-environment batches. Each completed batch checkpoints its original results and post-batch generator state. Replaying those results restores the correct generator position before the next live batch. Collection checkpoints only the original robot states, issued actions and tasks, plus the scene keys needed for replay. A killed in-flight batch may run again. The unrendered babble worker exits after each shard, as in the release.
- Fitting calls the released constructors, selected hyperparameters, episode split, loss, optimizer, early stopping and cache loader. Model weights initialize before upstream seeds its fit loop. The transport saves the actual pre-construction CPU RNG for each model. The published model directory includes weights, spec and held-out fit metrics. The smallest fit restart unit is one model. The released optimizer has no mid-epoch checkpoint, so interruption restarts the in-flight model from its saved initial RNG.
- The release’s VLA wrapper requests unversioned Hub metadata. The runtime owner’s offline resolver pins that request. The stock LeRobot loader can catch weight-load exceptions, so every lane requires its explicit all-keys-loaded confirmation and the pinned revision before collecting or scoring. Each allocation runs the repository GPU probe, maps EGL to CUDA, and records device names, PCI identities and per-lane logs. [Native jobs](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/eval/jobs.py) · [Model cache](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/models/cache.py)

## Qualification and first scheduler receipt

- On Mac, the release CLI produced exactly the runner’s 1,400 assignments per method and all 250 VLA collection jobs. CPU verification also matched all 500 native babble shard arguments, the 2/4/5-way partitions and the IQM for a 40-group fixture. [Mac receipt](evidence/reference-cpu-macos.json)
- On Baige CPU, the exact release environment fitted and reloaded four reduced-width test models on full-size synthetic pools. It recovered an interrupted collector and reproduced all five raw trace fields from uninterrupted native ReGround execution after restoring the CEM RNG. A separate native LIBERO test opened the 50 fixed states for each official task, completed one whole 2,000-target camera-free babble shard and ran a 280-step fixed-start native episode using a constant qualification action. The synthetic rates and constant-action episode are qualification evidence only. [Release CPU receipt](evidence/reference-cpu-release.json)
- The parent’s first two-Pro6K allocation, `job-jp5o2em8cal1` / `babble01`, passed the two-CUDA-device probe and mapped its two EGL devices. The training container failed to import OpenCV because the launcher supplied only one of the prepared system-library directories. It published zero shards and recorded 228 GPU-seconds (0.0633 GPU-hours) for the wrapper. The corrected launch pin above supplies both library directories. CPU readback confirmed that OpenCV mapped the prepared PCRE and GLib libraries. The full CPU check at the corrected SHA produced a byte-identical 2,046-transition shard (12 episodes, 45.49 s) and a complete 280-step test episode. [Corrected-pin CPU receipt](evidence/reference-cpu-release-23a66415.json)

- A separate 2,800-row fixture passed the production CSV/trace checks and summary writer. Both methods matched the stock `main.py summarize` output on every released field, including the bootstrap endpoints. A flipped unsafe flag was rejected. The reservation test rejected a second allocation that would double-spend an outstanding reservation, then admitted it after reading the first attempt’s actual exit charge. [Summary and budget receipt](evidence/reference-summary-cpu.json) · [Executable check](reference_summary_cpu.py)

To repeat the data, model and simulator checks on the CPU host, source the shared runtime and run the command below. The script uses temporary run-local fixtures and writes a separate receipt.

```bash
source "$root/runtime/activate.sh"
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$root/env/bin/python" -B "$runner/reference_cpu.py" --native --simulator \
  --output "$root/reference-cpu-readback.json"
```

At the 15:17 Shanghai handoff, the parent had started corrected-pin attempt `babble03` on Baige job `job-rmvtf5i2znts` with four requested cards and a 5.93 GPU-hour reservation. The shared store had zero published production shards at that check. The stock π₀.₅ loader had confirmed every key on CPU. The development-GPU policy inference receipt was still pending. The parent’s next check is the allocation probe and first published shard. [Handoff receipt](evidence/reference-handoff.json)

## Inspect and resume

```bash
source "$root/runtime/activate.sh"
"$root/env/bin/python" -B "$runner/reference.py" status
tail -n 4 "$root"/reference-native/attempts/collect01/collect-gpu*.jsonl
# Run on CPU once all reference shards are complete.
"$root/env/bin/python" -B "$runner/reference.py" summarize
```

Each attempt contains `reservation.json`, `gpu.json`, `launch.log`, append-only per-lane events and logs, and an exit receipt when the wrapper can write one. Each published event records elapsed seconds, units and measured rates. Per-batch checkpoints and the active lane log show progress before the first shard completes. Check first publication, then every 20 minutes. Use the scheduler’s own termination receipt to reconcile an abrupt exit. Resume with the same science inputs and a new attempt ID. GPU phase timing comes from the assigned job. CPU fixture timing is only a qualification measurement.

## Git deployment

The Baige checkout at `$root/src-$pin` already contains the tested files. To deploy the same revision into a fresh detached checkout, the following bundle needs the exact base commit shown below. The run-local bare repository already contains it. This procedure uses the Mac main clone and preserves every active checkout.

```bash
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
pin=23a6641562a4d833837f27ab0535a20ad94e8ec8
base=952071c22600a7cc1bcfa819dac801996931a70a
git merge-base --is-ancestor "$pin" main
git bundle create "$run/reference-$pin.bundle" main "^$base"
scp "$run/reference-$pin.bundle" "baige-dev:$root/"
ssh baige-dev bash -s -- "$root" "$pin" <<'REMOTE'
set -eu
root=$1 pin=$2
git -C "$root/source.git" bundle verify "$root/reference-$pin.bundle"
git -C "$root/source.git" fetch "$root/reference-$pin.bundle" main
if [ ! -d "$root/src-$pin" ]; then
    git -C "$root/source.git" worktree add --detach "$root/src-$pin" "$pin"
fi
test "$(git -C "$root/src-$pin" rev-parse HEAD)" = "$pin"
REMOTE
```
