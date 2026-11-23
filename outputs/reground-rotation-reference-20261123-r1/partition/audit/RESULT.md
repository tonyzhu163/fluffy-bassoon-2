# Cross-host calibration publication audit

**At 16:10:04 Shanghai on November 23, 2026, two Baige job hosts had each reported publishing the same 4 policy collection keys and the same 101 motor-babble keys.** All 105 pairs agreed on transitions and episodes. Every surviving NPZ and JSON pair passed the file checks. Counting both receipts would credit an extra 19,638 collect transitions across 92 episodes and an extra 210,151 babble transitions across 1,091 episodes. All 105 pairs had overlapping start-to-publication intervals on the recorded host clocks. [Every key, both owners, counts and current file hashes](duplicate-keys.tsv) · [Counts by attempt](counts.json)

| Phase | Publication receipts from all lanes | Distinct published keys | Keys with two publishers | Other keys started on two hosts |
|---|---:|---:|---:|---:|
| Collect | 64 | 60 | 4 | 0 |
| Babble | 453 | 352 | 101 | 6 |

A key consists of seed, shift, level, phase and shard. A comparable pair requires a newline-terminated `published` event from each of two distinct reservation hosts for the exact key by the cutoff. Lane logs echo some events. Each event was counted only from its JSONL stream. The TSV covers all 111 keys with evidence from two hosts, ordered by phase and key. A zero in its second column identifies the six keys with insufficient publication receipts. [Timestamped receipts, host records and log excerpts](receipts.json)

## Publishers and coverage

| Attempt | Baige job ID | Starts | Publications | Source pin |
|---|---|---:|---:|---|
| collect01 | job-3ohhwfrwt34h | 54 | 50 | 23a66415 |
| collect-seed0-01 | job-hr51yhgxiogv | 16 | 14 | 23a66415 |
| babble03 | job-rmvtf5i2znts | 243 | 239 | 23a66415 |
| babblecpu01 | job-0uwxvy2croqa | 222 | 214 | 516a2f75 |
| babble01 | job-jp5o2em8cal1 | 2 | 0 | 12df9436 |

The four publishing attempts' reservation hosts match their saved probe hostnames. The original babble01 start on seeds 0 and 1, nominal shard000, preceded the failed simulator import. Each key later received one babble03 publication. Of the other four cross-host starts, three had one publication at cutoff and seed2/none/0/babble/shard035 had none. It also had no NPZ or sidecar at file readback. The local execution record lists submissions for babble02 and collect-s1to4-01 (job-v3wxgfwqi477). Neither had a production attempt directory in this snapshot. The 15:39 local speed snapshot preserved 68 babble03 and 4 collect01 publication rows. All 72 matched the remote streams, including counts and timestamps. The eight local speed qualification fixtures used separate output trees and are excluded. [Reservations and stream inventory](receipts.json) · [Local snapshot comparison](survivors.json) · [Submission record](../../EXECUTION.md)

## Collect checkpoints and final owners

All four keys are seed 0. Batch indices below start at 0. The owner columns give each lane's recorded action, then the last sidecar's named publisher. The current sidecar publisher agreed with the later publication receipt in every case.

| Cell, shard | Transitions / episodes | collect01 | collect-seed0-01 | Current sidecar |
|---|---:|---|---|---|
| none/0, 001 | 4,341 / 32 | GPU 2 wrote 0–7 | GPU 0 replayed 0–7 | collect-seed0-01, PID 1462 |
| none/0, 003 | 4,257 / 28 | GPU 2 replayed 0–1, wrote 2–6 | GPU 0 wrote 0–6 | collect01, PID 2867 |
| rot_z/90, 001 | 5,520 / 16 | GPU 3 wrote 0–3 | GPU 1 replayed 0–3 | collect-seed0-01, PID 1463 |
| rot_z/90, 002 | 5,520 / 16 | GPU 1 wrote 0–3 | GPU 1 replayed 0 and 2, wrote 1 and 3 | collect01, PID 2866 |

For each of the **23** surviving batch checkpoints, the audit verified ZIP integrity, consecutive index, release and policy revisions, all four expected task/reset/replica/scene-seed tuples, trace dimensions and finite values. The cumulative transition count at each batch boundary matched both lane logs on all 23. Reassembling the current batch traces in native order reproduced every byte of each final shard's four arrays (`X`, `A`, `lens`, `task`). Native NumPy recompression in memory also reproduced the complete compressed-file SHA-256 for **4/4** shards. [Checkpoint and final hashes, full metadata, episode identities, counts and log line references](survivors.json)

Sixteen batch paths had one writer and one replay. Both jobs logged a fresh write to the other seven. On nominal shard003, the surviving archives for batches 0–1 carry the specialist's PID 1462, and batches 2–6 carry the original collector's PID 2867. On rotated shard002, batch 1 carries specialist PID 1463 and batches 0, 2 and 3 carry original collector PID 2866. Each PID in the ZIP member prefix mapped uniquely to a logged writer using independent collect sidecar host, PID and rank fields. This establishes mixed checkpoint provenance in two valid current shards. Full replay on the two shard001 keys explains how counts and payloads can agree without an independent second rollout. [Per-batch ownership evidence](survivors.json)

## Final-file checks and byte identity

The Baige CPU readback ran from 16:10:05 to 16:10:12 Shanghai. For all 105 comparable keys it read the full compressed archive and JSON sidecar, hashed both, checked ZIP CRCs and loaded the four arrays. It checked finite float32 states and actions, positive int64 episode lengths, the state/action shapes, the released task order and the stopping boundary (2,000 transitions with a final whole episode for babble, 4,000 with a final four-episode batch for collect). All sidecar transition counts, seeds and cells matched. All collect sidecars had the pinned policy and release and `x0_fixed=true`. All babble sidecars had the expected shard index, task list and native babble configuration. File identity, size, modification time and change time were stable across each key's read. The five additional keys with a single publication receipt also had readable, valid surviving pairs. [Checks and exact hashes by key](survivors.json)

The publication receipts contain counts and timings but no per-writer payload digest. The inspected attempt-specific shard staging directories retained no files for the 105 comparable keys. Shared checkpoint paths retain a single version per batch. Consequently no pair of separately retained production NPZs establishes historical writer-to-writer byte identity. For the two fully replayed collect keys, the sole writer's checkpoints and both logs support a common input sequence. For the seven independently written batches the overwritten checkpoints are unavailable. Collect sidecars record publisher-specific host, PID and command line. Babble sidecars record a runtime and scientific fields but omit an owner. The TSV gives both babble receipt owners and leaves the current metadata-owner field empty. Matching counts, timing proximity and the earlier small-fixture hashes cannot identify the historical babble bytes or attribute the surviving babble payload to a host. [Receipt fields](receipts.json) · [Staging inventory and current sidecars](survivors.json)

## Shared paths and temporary files

The production claim used a nonblocking exclusive `flock` on the key's shared lock path and rechecked for an existing NPZ after taking it. The two publication receipts per key establish that this protocol admitted duplicate publication on these allocations. The reader saw GPFS at the lock directory. The data do not isolate locking semantics, cross-host visibility of publication, or the precise order of filesystem calls across nodes. [Pinned claim and lane code](../../reference.py) · [Mount and overlapping events](receipts.json)

Collect wrote each checkpoint to `batches/<key>/batchNNNN.<pid>.tmp` and replaced the shared `batchNNNN.pt`. Seven same-destination writes are confirmed. The two participating lane PIDs differed for each dual-written batch. Their temporary paths therefore differed under the pinned code. PID-only names can coincide if different host namespaces assign the same PID to writers working on the same key and batch. The scan found **zero residual temporary files** under the production data, batch and relevant attempt staging trees, and the captured logs contained no temp-related exception. The scan cannot reconstruct a transient rename or an already removed temporary. [Archive PID mapping and scanned roots](survivors.json) · [Production log excerpts](receipts.json)

The native NPZ and JSON temporaries for shard assembly lived inside separate attempt directories. Both publishing paths then replaced the canonical JSON, followed by the canonical NPZ. Each replacement is a separate operation. The observed coherent current pairs do not establish that a reader always saw same-writer metadata and payload together. A sidecar's host identifies its author. The NPZ itself carries no host identifier. [Reference publisher](../../reference.py) · [CPU babble publisher](../../speed/babble.py)

The audit used Linux CPU on ssh baige-dev, consumed 0 GPU-hours and launched no jobs. Receipt totals use a fixed cutoff while production continued. The live reads, retained snapshot and reconstruction commands are documented in [REPRODUCE.md](REPRODUCE.md).
