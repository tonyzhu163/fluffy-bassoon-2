# Eight CPU workers for registered ReGround babbling

Use one RTX 4090 Distributed Training master with eight allocated CPUs and eight camera-free native workers when calibration shards remain. The matched eight-shard CPU fixture took **23.73 s** with the new transport and **82.10 s** with four independent registered lanes. Both started with an empty private Numba cache. Every compressed NPZ and every native metadata field except elapsed time matched the serial reference. This investigation used **0 worker GPU-hours**. The parent owns submission and scheduling. [CPU receipts and shard hashes](qualification.json) · [Launcher](launch.sh)

## What costs time

At 15:39 Shanghai on November 23, the four-card `babble03` attempt had published 68 of the 500 registered shards. Each rank's first shard took 455.7–458.3 s. The next 64 took 35.2–81.6 s, median 40.1 s. A rank runs one camera-free child at a time. In an initial CPU import probe, importing the native task resolver took 0.09 s without loading Torch. After Torch was loaded, importing `lerobot.envs.libero` took 67.5 s in a fresh process before any simulator existed. On the warmed development machine the whole transport preload took 5.5 s in one trial. Import and storage costs vary substantially with cache state. The release imports Torch through LeRobot's simulator adapter and through the VLA module used by the rollout workers. [Time-stamped active job events](active-snapshot.json) · [Registered lane and worker](../reference.py)

The cheaper experiment used the registered input arguments with only the requested count reduced from 2,000 to one transition. The release finishes the last whole babbling episode. Shards 0–7 covered all eight tasks, both bodies and all five adaptation seeds, for 1,547 actual transitions per arm. Workers used the native default configuration, resets, physics, stopping condition, encoder and validator. The fixture output root was private. The dev cgroup afforded eight CPUs and 32 GiB even though its affinity mask showed 64 CPUs.

| Baige CPU fixture, same eight shard keys | Workers | Cache at start | Wall seconds |
| --- | ---: | --- | ---: |
| Registered `perform`, sequential | 1 | Existing shared Numba cache | 78.70 |
| One imported CPU parent, fresh child per shard | 8 | Existing shared Numba cache | 12.20 |
| Four independent registered `perform` lanes | 4 | Fresh private Numba cache | 82.10 |
| One imported CPU parent, fresh child per shard | 8 | Fresh private Numba cache | 23.73 |

The last comparison gives **3.46×** the throughput on this fixture. Timing starts after the plan exists and includes worker creation and publication. The eight-worker figures include preload and final shard validation. The four-lane control also starts its four independent Python lanes inside the timer. Each arm ran once on shared infrastructure. These short shards emphasize process startup. Full-size throughput on a one-card scheduler host, performance at 9–24 workers and the contribution of each individual optimization remain unmeasured. The largest child RSS in the private-cache trial was 2.43 GiB, including shared pages. [Fixture and controls](qualify.py) · [Raw timings, cgroup limits and worker events](qualification.json)

## Publication and recovery

The launcher reads the 500 jobs directly from the detached registered runner's `units("babble", SEEDS)` in an isolated planner. A separate, single-threaded Linux parent preloads the verified native simulator modules once. It creates each simulator in a new child and reaps that child after one shard. CUDA is hidden from the CPU payload after the allocation probe. The parent rejects initialized CUDA/EGL or multiple native threads before forking. Each child calls the registered `claim`, `babble_child`, `output` and `check_pool_shard`. The existing nonblocking filesystem lock covers validation and both renames. The native JSON is published first. Renaming the NPZ commits the shard. [CPU dispatcher](babble.py)

A private fixture placed a stale attempt directory and a final JSON without its NPZ beside a valid shard. A separate process held the registered lock on the missing shard. The transport reported that shard outstanding. After release, a fresh attempt reproduced its native hash and preserved the previous NPZ's modification time. Each invocation requires a fresh attempt ID. A sweep verifies all visible final pairs. Exit 75 and `summary.json` list any outstanding units claimed elsewhere at the end of the sweep. A worker or validation error exits nonzero. An interrupted attempt's private partials can be left in place. [Recovery and ledger checks](qualification.json)

## Parent deployment

Use the run's qualified image, exclusive RTX 4090 queue and existing read/write PFS mount at `/mowenjie`. Request one card, eight vCPUs and 32 GiB if available in the queue. The registered allocation supplied 40 CPUs for four cards. The 8–24 argument controls CPU concurrency and must fit the queue's accepted per-card request and the container's effective CPU quota. The launcher checks affinity and cgroup limits, and rejected nine workers on the eight-CPU host. A dedicated CPU-only Distributed Training queue and its native image were not qualified in this investigation.

Only the transport needs a new detached checkout. It imports the reference and GPU probe from the existing `src-23a6641562a4d833837f27ab0535a20ad94e8ec8` checkout, runs the existing `env/bin/python`, and uses the released upstream commit `16a98486f560676b9036d1a6d067b9dd51a5917a`. Startup verifies the pinned runtime receipts, interpreter and activation bytes, 150 installed distribution versions, the configured asset link and all 586 release asset files. Each future attempt has its own Numba cache. The Linux CPU readback passed those checks. [Verifier](check_runtime.py) · [Qualified identities](runtime.sha256) · [Runtime provenance](../RUNTIME.md)

Use the transport commit from the handoff for `TRANSPORT_SHA`. The example reserves at most 1,800 GPU-seconds (0.50 GPU-hours) against the parent's existing reference ceiling. The scheduler cap should cover launcher wall plus container overhead. Count all submitted allocations, including startup/teardown and other study arms, against the combined 50 GPU-hour budget.

```bash
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
pin=${TRANSPORT_SHA:?set the detached transport commit}
runner="$root/speed/src-$pin/$run/speed"
# Supply the parent's current cumulative reference ceiling, e.g. 40 under the registered split.
export REFERENCE_CAP_GPUH=${REFERENCE_CAP_GPUH:?set the cumulative ceiling}
bash "$runner/launch.sh" "$pin" babblecpu01 8 1800
```

All reservations use the registered `reference-native/attempts/.budget.lock`. Outstanding reservations are charged at their maximum and completed attempts at their recorded gross GPU-seconds. The budget test rejected 3,800 GPU-seconds against a 3,600-second fixture ceiling, then admitted the same 600-second single-card request after the earlier attempt reconciled to 2,800 seconds. An illustrative scale calculation for the 432 shards outstanding at the snapshot is `432 × 40.1 / 8 / 3600 ≈ 0.60` GPU-hours on one card, **if** eight CPU workers sustain that historical median simultaneously. It excludes qualification, cold startup, contention, overshoot variation and scheduler overhead. A first scored publication and subsequent steady-state lane events would supply the actual rate. The one-card GPU probe and scored path have not been run by this worker.

## VLA collection and test evaluation

The existing 636.9 s cold RTX 4090 smoke includes setup, strict load, prediction and one step. In the active four-card collector each rank logged a 362 s native policy load. Its first complete shard took 987–1,004 s including cold startup. The first rank's subsequent four-episode checkpoint gaps within that shard were about 22–43 s. The registered worker retains the strictly loaded policy across shards. A fresh task assignment respawns its four simulator processes. These measurements cover different operations. [GPU smoke](../runtime/qualification/dev-gpu-smoke.json) · [Active job snapshot](active-snapshot.json)

| CPU-only transport check | Shared before | Verified node-local copy | Shared after |
| --- | ---: | ---: | ---: |
| Complete strict policy loader process | 109.1 s | 130.1 s | 111.2 s |
| Four fresh native workers on a task switch, two switches | 10.97 / 10.08 s | 9.93 / 9.96 s | 9.70 / 9.90 s |

Copying and SHA-256-verifying the 16 model/tokenizer snapshot files (7.495 GB) took another 19.7 s. Each CPU policy process passed the registered strict-load and revision checks. In the separate respawn trial, copying and comparing 4,611 existing Python cache files (99.4 MB) took 8.2 s. The interpreter, packages, assets, task order and native spawn path still came from the qualified runtime. A repeated same-task camera-free reset took 0.57–0.64 s. Switching tasks has a measurable cost. Neither tested copy produced a consistent speedup against both shared controls. Keep the registered VLA collection and evaluation transport. Four-slot CPU resets omit rendering, inference and trajectory collection, so the CPU switch timings do not decompose a GPU batch. [Loader and reset measurements](startup-cpu.json) · [Bytecode comparison](respawn-io.json) · [CPU probe](startup_cpu.py)

To repeat the fixtures on `baige-dev`, source the detached transport's `environment.sh`, set `CUDA_VISIBLE_DEVICES=""` and use a new directory from `mktemp -d "$root/speed/qualification.XXXXXXXX"`. Run `"$py" -Bu qualify.py serial "$q"` and then `parallel` with the existing qualified cache. For `private` and `four`, set `NUMBA_CACHE_DIR` to a distinct empty directory under `$q` for each arm. Run `resume` and `budget` against the same private fixture root. The standalone `startup_cpu.py measure --output FILE` and `bytecode --output FILE` commands use the same interpreter and hidden CUDA. The receipt records the Baige fixture directory, file identities and measured cgroup limits.
