# Native ReGround read transport

**Keep the current attempts on their qualified runtime.** For a fresh collection, Base, ReGround or streaming-reader attempt, the parent can opt into a single container-local copy of Transformers 5.5.4. It removes 2,720 shared model-source open requests from a fresh native simulator import. Warm Baige CPU timings show no consistent wall-time gain. A cold 4090 speedup has not been measured. [CPU receipt](transport/qualification.json) · [Optional launcher patch](transport/launchers.patch)

## What production measured

| RTX 4090 measurement | Observed wall time | What the clock covers |
| --- | ---: | --- |
| Development policy action | 636.88 s inside the smoke, 726 s allocation | Construct the environment, strict policy load, render, query and one simulator step. The smoke's internal timer starts after its top-level imports. |
| Collection `collect01`, lane 0 | 362 s once | Native `VLA.load_s` starts **after** imports of the policy class. It includes config, weights, device transfer and processors. |
| Same lane, first six four-episode batches | 107, 29, 22, 27, 23, 27 s | Differences of cumulative collection times. The native eight-task rotation and clean task-change spawns are active. Timings include policy, physics, rendering and checkpointing. |
| Base `base01`, lane 0 | 361 s once, then 131, 30, 31, 31 s | Strict load followed by the four native batches of a 16-episode shard. The lane published 8,320 controls in 775.72 s from its unit start. |
| Babble `babble03`, lane 0 | 455.68 s for 2,063 transitions | A separate first-shard clock. |

The source read was at 15:36 Shanghai on November 23. The lane logs are under `$root/reference-native/attempts/{collect01,base01,babble03}`. The one-time policy load and the recurring clean-spawn cost belong to different clocks. The observed 22–29 s later collection batches already bound how much time a faster import could remove from any of **those** batches. [Run accounting](EXECUTION.md) · [GPU smoke](runtime/qualification/dev-gpu-smoke.json) · [Release runner](REFERENCE.md)

## Cache contents and evidence

Transformers recursively reads model source to build its lazy import map, then globs image processors. Its package wheel is already in `$root/wheels` on project storage. The audit compared the installed package's file set and every file's bytes with the wheel. All 2,271 Python files and one type marker matched. The wrapper reads the 10,236,696-byte wheel sequentially, checks its pinned SHA-256, extracts only the package into private `/tmp` and reads back every file. The 7.47 GB policy, interpreter, distribution metadata, native libraries, LIBERO assets, JAX reader, canonical caches and run outputs use their existing project paths. [Wheel audit and staging](transport/wheel.py) · [Runtime pins](RUNTIME.md)

On the CPU host, one baseline process took 22.30 s through construction, reset and close of a two-camera spatial environment, with 14.64 s in construction and reset. The import profiler attributed 5.38 s cumulative to `lerobot.envs.libero`. Eight subsequent fresh processes ran the same worker import in shared/scratch/scratch/shared order twice, at low priority on one CPU. The table gives process wall seconds, including Python startup and exit.

| Source | Four process times (s) | Requests for Transformers model Python files per process |
| --- | --- | --- |
| Shared installation | 44.21, 5.28, 5.63, 5.44 | 2,720 on PFS across 1,355 distinct paths |
| Staged wheel | 6.14, 5.12, 6.51, 4.86 | 2,732 on scratch across 1,358 distinct paths, zero under the shared package |

The shared 44.21 s observation came first during changing CPU load. Filesystem caches were allowed to warm naturally. Neither that outlier nor the single end-to-end fixture pair estimates a causal cold-start gain. Staging took about 0.37 s on this host. It used 48,533,504 allocated bytes and 2,740 package inodes plus the private root directory. The expanded payload is 41,561,144 bytes. Each exclusive container pays that cost once for all ranks and their children. [Alternating import probe](transport/probe.py) · [Times and free-space receipts](transport/qualification.json)

On the same Baige CPU host, the stock and staged matched runs both completed `spatial_0`, start 15, scene seed 0, noise seed 709751874, with the archived seed-41 reader and the existing noise-driven fixture policy. All five NPY trace payloads matched byte for byte, including the 281 states, 280 proposals and issued actions, speeds and reference field. All 28 queries, 24 updates, camera hashes, non-timing CSV fields and native replay outcomes matched. The two complete wiring checks took 74.78 and 57.02 s in that order. The qualification uses one episode. The production runner still owns the four-slot batch, spawn, policy noise, CEM, control and termination logic. [Trace and ledger comparison](transport/compare_episode.py) · [Matched harness](MATCHED.md)

The strict policy check ran the stock `runtime/qualify.py` on two 256 × 256 CPU-rendered views of `spatial_0`, fixed start 0, scene/noise seed 0, chunk 0. Both processes confirmed all pretrained keys, generated the identical 1 × 50 × 7 float action chunk (SHA-256 `5d59ddddd2b4c0fb42626ef707a3bea36769b66f8e0b368706017d4d86478fbd`), and applied its first action with identical state-change norm. The staged and fresh stock processes took 149.92 and 215.58 s, respectively, under changing CPU contention. The earlier runtime smoke has a different chunk hash. Its CPU settings were not paired with this comparison. The present conclusion is confined to the two fresh processes with identical settings. [Strict policy and runtime records](transport/qualification.json)

The raw logs, episode outputs and receipts are on `baige-dev` in `$root/transport/cpu-PeIkmV/`. The native and matched harness came from `cab855a03c8ff1a3f1926513369099bf11c4b859`, upstream from `16a98486f560676b9036d1a6d067b9dd51a5917a`. The policy and simulator tests used the staging implementation from `d666717b`; `948e3a53` added the alternating probe. The final wrapper at `0a5fbddd` adds the runtime-root guard. Its successful staging readback and wrong-root rejection were checked separately. Private qualification scratch was removed. At close, `/tmp` had 31,503,069,184 bytes and 3,132,519 inodes available. Project storage had 258,109,079,552 bytes and 958,583 inodes available while other jobs continued writing. The shared project already held the wheel. The retained transport scripts and run evidence occupy less than 1 MiB on Baige. [Storage snapshots and source digests](transport/qualification.json)

## Apply to a fresh attempt

The patch was checked against the `cab855a0` launchers. Apply it in the parent's **new integration checkout** containing this transport directory. Commit the launchers, deploy that pin into a fresh detached checkout using the existing Git procedure, and use the new pin and a fresh attempt ID. The production launcher checks its own Git identity. The code-only transport handoff is also on Baige at `$root/transport/tool-0a5fbddd9972e77d351ac8229ec18660515e0fb7/`. [Git deployment and attempt arguments](REFERENCE.md) · [Streaming evaluation](stream.sh)

```bash
run=outputs/reground-rotation-reference-20261123-r1
git apply --check "$run/transport/launchers.patch"
git apply "$run/transport/launchers.patch"
git commit --only -m 'Enable optional native ReGround package staging' -- \
  "$run/reference.sh" "$run/stream.sh" "$run/symmeta_adapter.sh"
```

On Baige, re-audit after any runtime rebuild. Use the existing approved arguments and caps for the job. Prefix either normal launcher command with the opt-in flag.

```bash
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
runner="$root/src-$pin/$run"
/usr/bin/python3 -IB "$runner/transport/wheel.py" --audit-install

# In an already approved allocation, using a fresh attempt ID.
REGROUND_READ_CACHE=1 bash "$runner/reference.sh" "$pin" "$id" "$cards" "$wall" collect
# For streaming, use its own fresh attempt, scheduler cap and selected seeds/shards.
REGROUND_READ_CACHE=1 bash "$runner/stream.sh" "$pin" "$id" "$cards" "$wall"
```

The patch stages inside the existing reservation and timeout. It reapplies the scratch import path after the stock activation script sets `PYTHONPATH`. Native policy and simulator descendants inherit it. The archived reader deliberately builds its own Python 3.11 path. The run's offline resolver and package metadata still come from their original locations. Standalone CPU commands can run after activation as `bash "$runner/transport/with-cache.sh" "$root/env/bin/python" -B ...`. The wrapper emits a staging receipt to the attempt's launch log. [Wrapper](transport/with-cache.sh)

A digest, capacity or extraction failure exits before starting the payload. The parent can omit `REGROUND_READ_CACHE` on a fresh attempt and resume the ordinary sealed batches. Normal completion and payload failure remove the wrapper's private directory. A forced container teardown discards its local scratch. A deliberately failing CPU payload returned its original status 17 and its directory was absent afterward. [Recovery check and provenance](transport/qualification.json)
