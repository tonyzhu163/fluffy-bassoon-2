# Progress-selected overlap on two substituted arms

**Stop this intervention on the registered panel.** On the 16 matched arm/task cases, the displacement-selected overlap succeeded 12 times, the raw frozen policy 16, the tuned policy 14, and fixed overlap 10. Selecting the donor phase recovered three fixed failures and lost one fixed success. All three recoveries already succeeded under both raw and tuned. The registered condition for fresh-layout confirmation required favorable native success against all three controls. [Registration](PLAN.md) · [Paired score](evidence/native01/paired-layout31.json)

## Native results

All 64 episodes reached an endpoint. Each successful episode had zero rewards until a single terminal reward of one. Each failure had zero return and truncated at the official horizon. The table gives native success for both arms on each released row-31 BASE task, in the registered order. The last column gives generalized position and velocity dimensions of the compiled model, equal across arms within each task. [Per-episode audit](evidence/native01/audit.json) · [Original episode JSON, compressed](evidence/native01/episodes.tar.gz)

| Task | IIWA raw | Tuned | Fixed | Progress | Kinova3 raw | Tuned | Fixed | Progress | nq / nv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| libero_10 / 5 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 23 / 21 |
| libero_10 / 8 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 24 / 22 |
| libero_goal / 3 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 41 / 37 |
| libero_goal / 6 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 41 / 37 |
| libero_object / 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 58 / 51 |
| libero_object / 8 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 58 / 51 |
| libero_spatial / 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 48 / 43 |
| libero_spatial / 8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 48 / 43 |
| **Successes / 8** | **8** | **7** | **6** | **6** | **8** | **7** | **4** | **6** | |

Each interval is the exact central 95% task-bootstrap percentile interval, in percentage points (pp). A pooled draw resamples the same task identity for both arms with equal arm weight. An independent enumeration of all 6,435 unordered eight-task samples, weighted by their multiplicities among 16,777,216 ordered draws, reproduced every count and endpoint of the intervals. There is one layout and one noise bank per task. The intervals describe variation across these eight tasks. [Paired score](evidence/native01/paired-layout31.json) · [Independent computation](native_readback.py)

| Panel | First mode minus second | Rescues / second-mode failures | Losses / second-mode successes | Ties | Difference pp [95% interval] |
|---|---|---:|---:|---:|---|
| IIWA | Progress − raw | 0 / 0 | 2 / 8 | 6 | −25 [−62.5, 0] |
| IIWA | Progress − tuned | 0 / 1 | 1 / 7 | 7 | −12.5 [−37.5, 0] |
| IIWA | Progress − fixed | 1 / 2 | 1 / 6 | 6 | 0 [−37.5, 37.5] |
| IIWA | Fixed − tuned | 0 / 1 | 1 / 7 | 7 | −12.5 [−37.5, 0] |
| Kinova3 | Progress − raw | 0 / 0 | 2 / 8 | 6 | −25 [−62.5, 0] |
| Kinova3 | Progress − tuned | 0 / 1 | 1 / 7 | 7 | −12.5 [−37.5, 0] |
| Kinova3 | Progress − fixed | 2 / 4 | 0 / 4 | 6 | +25 [0, 62.5] |
| Kinova3 | Fixed − tuned | 0 / 1 | 3 / 7 | 5 | −37.5 [−75, −12.5] |
| Both | Progress − raw | 0 / 0 | 4 / 16 | 12 | −25 [−50, −6.25] |
| Both | Progress − tuned | 0 / 2 | 2 / 14 | 14 | −12.5 [−37.5, 0] |
| Both | Progress − fixed | 3 / 6 | 1 / 10 | 12 | +12.5 [−12.5, +37.5] |
| Both | Fixed − tuned | 0 / 2 | 4 / 14 | 12 | −25 [−43.75, −6.25] |

## How often the controller changed a command

A boundary is a policy query following a complete five-control block. At each eligible control, the same-history comparison uses that mode's actual current and preceding original draws. It isolates the change in the issued action from choosing a different donor index. The endpoint comparison uses the independently reset closed-loop episodes. [Per-task exposure and phase histograms](evidence/native01/paired-layout31.json)

| Arm and mode | Eligible boundaries | Selected index counts, j = 0, 1, 2, 3, 4, 5 | Departures from 5 | Eligible controls | Changed canonical / sent vs tuned on same history | Changed sent vs fixed on same history |
|---|---:|---|---:|---:|---:|---:|
| IIWA fixed | 353 | 0, 0, 0, 0, 0, 353 | 0 | 1,752 | 1,752 / 1,752 | 0 |
| IIWA progress | 367 | 96, 21, 12, 18, 49, 171 | 196 | 1,830 | 1,830 / 1,830 | 980 |
| Kinova3 fixed | 387 | 0, 0, 0, 0, 0, 387 | 0 | 1,929 | 1,929 / 1,929 | 0 |
| Kinova3 progress | 344 | 92, 16, 13, 26, 28, 169 | 175 | 1,704 | 1,704 / 1,704 | 864 |

Progress departed from index five at 371 of 711 decisions and changed 1,844 of 3,534 eligible issued controls relative to fixed on the same history. Every arm/task saw at least one departure. Fixed overlap itself lost four tuned successes. Choosing the phase gave a net recovery against fixed on Kinova3 and exchanged one success for one failure on IIWA. Raw already succeeded on every task. The outcomes do not identify which individual command or contact caused a terminal difference. The claim is limited to these eight task identities under nominal commands, including the previously exercised Kinova3 drawer layout. [Registration and prior exposure](PLAN.md)

## Pairing and physical identity

- Within all 16 quartets, the complete environment, physical state, public reset, compiled model, first policy input, original opening proposal and noise bank matched across the four modes. All 64 resets were accepted on their first creation attempt. The three corrected modes issued identical opening commands. Every per-query noise digest matched the regenerated bank. The policy drew ten seven-coordinate actions per query using a 10 × 32 latent noise slice and replanned every five controls. The official horizons were 520, 300, 280 and 220 controls for the 10, goal, object and spatial suites. [Paired score](evidence/native01/paired-layout31.json) · [Episode audit](evidence/native01/audit.json)
- Across bodies, each task matched the released row-31 fingerprint and BDDL digest in the release receipt. Both bodies also matched in pre-settle scene fingerprint, perturbation plan, prompt, horizon, query seed, noise seed and bank digest. Body-specific full physics and settled pose hashes differed on all eight tasks. The native reset measures object poses against the same settled Panda donor. The largest recorded discrepancy from that donor was 3.14 × 10⁻¹⁵ m in position and 4.22 × 10⁻¹⁵ rad in rotation. Each quartet retained its own body's exact settled reset and public input. [Release rows](evidence/release-banks.json) · [Cross-body fingerprints and pose errors](evidence/native01/audit.json)
- On every task, six compiled-model fields differed between bodies. They identify the arm, model class, joint names, joint links, arm meshes and compiled XML digest. Both bodies used seven arm joints, identical PandaGripper XML, a relative 20 Hz OSC_POSE controller, and the same camera and TCP specifications. The object suite used OnTheGroundIIWA and OnTheGroundKinova3 with NullMount. The remaining suites used the respective Mounted classes with RethinkMount. All episodes shared the frozen weight manifest, source-law digest and registered PR7 prediction export. [Physical model records](evidence/native01/episodes.tar.gz) · [Weights and source identities](evidence/weights.json)

## Why the first development readback failed

The worker-side pinned readers each exited zero on 32 episodes. On the Baige development host, the default OpenBLAS kernel was Cooperlake. Replaying the recorded output plans with that kernel produced 638 X and 2,722 Y coordinate differences on 13,288 executed controls. The largest absolute differences were 5.96 × 10⁻⁸ and 1.19 × 10⁻⁷ respectively. The other five coordinates agreed byte for byte. Explicit Haswell, SkylakeX and Cooperlake dispatch gave the same discrepancy counts and maxima. Each stopped the full unchanged reader on issued-command equality. [Kernel comparisons for every control](evidence/native01/audit.log) · [First explicit failure](evidence/native01/cpu-pair02/reader.log)

With **OPENBLAS_CORETYPE=Sandybridge**, the original two-arm reader on development CPU 44 exited zero in fresh attempt cpu-pair05. Its two arm sections equal the complete worker receipts. The executable was Python 3.11.17 with NumPy 1.26.4 and OpenBLAS 0.3.23.dev on the AMD EPYC 9W24 development VM. The runtime used one BLAS thread, an empty CUDA visibility mask and JAX's CPU platform. The library reported the selected kernel directly. This identifies a dispatch setting that reproduces the recorded arithmetic. The records do not identify the workers' physical CPU models. [Exact runtime](evidence/native01/cpu-pair05/runtime.log) · [Exit receipt](evidence/native01/cpu-pair05/exit.txt) · [Worker IIWA](evidence/native01/IIWA/result.json) · [Worker Kinova3](evidence/native01/Kinova3/result.json)

A second pass, cpu-pair06, checked all 2,676 full output plans (187,320 float32 values). Explicit separate float32 products and sums for XY, with both native clipping stages, reproduced all 93,016 issued coordinates byte for byte. The largest issued difference was zero. Replaying the controller in its original reduction order also reproduced all 31,995 stored alignment values byte for byte, including donor commands, observed displacements, the six waypoints and distances. Both the nearest-waypoint rule with the registered tie break and the pinned reader recovered every recorded progress index. The audit independently checked the reward tapes, terminal flags and native motion summaries. The evaluator pin remains d70e206a00c6c5d59f190ec4a9b0bc6df2ec82c4. The earlier failed cpu-pair01 is intact. [Supplemental reader](native_readback.py) · [Audit log](evidence/native01/audit.log)

## Execution and evidence

| Arm | Baige Distributed Training worker | GPU | Start to end (UTC, 2026-11-24) | Launcher wall time |
|---|---|---|---|---:|
| IIWA | job-xrzslhwy0nni-master-0 | RTX 4090 | 02:56:52–03:12:26 | 934 s |
| Kinova3 | job-6cew0txsfs7d-master-0 | RTX 4090 | 02:57:06–03:12:26 | 920 s |

Both native jobs exited zero. Their combined launcher wall time was 0.515 single-GPU hours. This final readback used only the development CPU. The paired score, exit receipts, full worker logs, GPU probes, explicit kernel-attempt logs and audit were copied back from Baige. The portable 3,899,826-byte gzip tar archive contains all 64 original JSON files with their complete native reward and action tapes. Each archive member was checked against its source SHA-256 and all transferred file hashes matched on the receiving host. [Artifact hashes](evidence/native01/SHA256SUMS) · [Full IIWA log](evidence/native01/IIWA/run.log) · [Full Kinova3 log](evidence/native01/Kinova3/run.log)

To reproduce the paired score on Baige, run from the deployed pinned checkout with a fresh destination. The kernel-attempt script records the failed and successful attempts separately. The final command reruns the supplemental byte and endpoint comparisons. [Kernel-attempt script](native_readback.sh)

```bash
pin=d70e206a00c6c5d59f190ec4a9b0bc6df2ec82c4
d=/mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-d70e206a00c6/outputs/frozen-transfer-progress-20261124-r1
source "$d/runtime.sh"
out="$shared/cpu-pair07"
mkdir "$out" && (set -e
    runtime
    export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_CORETYPE=Sandybridge
    taskset -c 44 "$python" -Bu "$d/readback.py" --iiwa "$shared/gpu01-IIWA" \
        --kinova3 "$shared/gpu01-Kinova3" --out "$out/paired-layout31.json"
    PYTHONPATH="$d:$PYTHONPATH" taskset -c 44 "$python" -Bu "$shared/cpu-pair06/native_readback.py" \
        "$shared" --score "$out/paired-layout31.json" > "$out/audit.json"
)
```
