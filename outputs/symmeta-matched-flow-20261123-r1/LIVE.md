# Fresh UR5e π₀.₅ paired evaluation

The parent submitted Baige job `job-sjubp8q0tg1x` at 14:43 Shanghai on CPU-verified commit `08cf720efbb83231342fe22b285038065f254cda` for **96 fresh pairs across eight tasks**, including **64 held-out pairs**. The parent owns the scheduler and monitoring. This tooling handoff used CPU only. [Parent launch record](LAUNCH.md) · [Run plan](PLAN.md)

Two detached trees passed prepare on the actual inputs. The queued job uses `gpu01` and the `live-cpu01` reference panel with SHA-256 `276a0dce7685cefb4aad7898ccdb568d5e1fac46e19377bedc49b4e90b467cd2`. The subsequent deployment pin `d268f189c1bcd0a173b88936bf3ad403f9418594` also checks the committed offline receipt and adds its provenance to the panel. The 96 query rows, policy, rollout code and statistics are identical on both pins. [Frozen prediction receipt](evidence/prediction-provenance.json)

## Launch

The scheduler configuration is Baige **分布式训练**, one exclusive node, **2 RTX PRO 6000D**, **90 minutes**, automatic retries disabled, using the runtime image and shared mount from the qualified two-card FK job `job-pc0ioesm8tle`. This bounds the allocation at 3 GPU-hours. The command below reproduces the parent's invocation on its pinned tree and supplied run root.

```bash
bash -lc '
set -eu
run=symmeta-matched-flow-20261123-r1
root=/mowenjie/fluffy-bassoon-2/outputs/$run
pin=08cf720efbb83231342fe22b285038065f254cda
sha=c6828c9c848ccd0ce10bc89a5c91356b66c58619b08cbc07ba15d2af7608e191
exec bash "$root/live-src-$pin/outputs/$run/live.sh" run "$pin" "$root" gpu01 "$sha"
'
```

The queued job owns the fresh attempt name `gpu01`. For a separately authorized invocation of the later deployment, the same command takes `pin=d268f189c1bcd0a173b88936bf3ad403f9418594` and a fresh attempt name in place of `gpu01`.

The launcher creates its attempt directory exclusively. Inside the scheduler container it rechecks the sources, checkpoint bytes and all inputs, then runs `scripts/[train]gpu_probe.py --expected-count 2 --purpose evaluation --baige-container --output "$out/gpu.json"` from the pinned runtime. The probe checks scheduler GPU UUIDs, CUDA/PCI mapping, device occupancy and a real tensor operation on each card. The launcher checks the Pro6K device names. One CUDA-visible device and its corresponding EGL ordinal serve each lane. Lane 0 runs task shards 0, 2, 4, 6 and lane 1 runs 1, 3, 5, 7. JAX uses CUDA, MuJoCo uses NVIDIA EGL, and `NCCL_P2P_DISABLE=1` is job-local. The launcher’s wall limit covers prepare through merge.

Check the first complete pair on each lane in `gpu01/shard-0/00.json` and `gpu01/shard-1/12.json`, alongside `gpu01/gpu.json`. Then check progress every 20 minutes. Each finished file contains both native trajectories’ rewards, terminal flags, success, controls, full reset receipts and opening-proposal hashes. The throughput reference is 1,101 seconds for the previous two-arm FK job on two cards. Final acceptance requires exit code 0 in `exit.txt`, a valid `summary.json`, and exactly 96 complete pair files covering indices 0–95.

## Deployment and CPU prepare

The following command reproduces deployment from the Git bundle already staged in this run. It verifies the existing detached tree or creates it on the shared mount. The bundle contains the code and the frozen predictions. The command compares the external input to the Git blob before CPU preparation.

```bash
ssh baige-dev bash -s <<'BAIGE'
set -euo pipefail
run=symmeta-matched-flow-20261123-r1
root=/mowenjie/fluffy-bassoon-2/outputs/$run
pin=d268f189c1bcd0a173b88936bf3ad403f9418594
sha=c6828c9c848ccd0ce10bc89a5c91356b66c58619b08cbc07ba15d2af7608e191
bundle="$root/live-$pin.bundle"
tree="$root/live-src-$pin"
g="$root/live-source.git"
printf '%s  %s\n' 3b87646e003b9748bff98042cf65c9ae1b5695c0567ac54c5d9e58cbbd5bdfbc "$bundle" | sha256sum -c -
if [[ ! -d $g ]]; then
    git clone --bare --shared "$(dirname "$root")/symmeta-fk-ur5e-20261123-r1/git" "$g"
fi
if [[ ! -e $tree ]]; then
    git -C "$g" fetch --no-tags "$bundle" refs/heads/main
    git -C "$g" worktree add --detach --no-checkout "$tree" "$pin"
    git -C "$g" config extensions.worktreeConfig true
    git -C "$tree" config --worktree core.bare false
    git -C "$tree" sparse-checkout init --cone
    git -C "$tree" sparse-checkout set "outputs/$run" outputs/symmeta-fk-ur5e-20261123-r1 \
        outputs/symmeta-camera-command-20261123-r1 outputs/xemb-live-wave-20261121-r1
    git -C "$tree" checkout --detach "$pin"
fi
cd "$tree"
test "$(git rev-parse HEAD)" = "$pin"
test -z "$(git --no-optional-locks status --porcelain)"
mkdir -p "$root/evidence"
if [[ ! -e $root/evidence/predictions.json ]]; then
    ln "outputs/$run/evidence/predictions.json" "$root/evidence/predictions.json"
fi
git show "$pin:outputs/$run/evidence/predictions.json" | cmp - "$root/evidence/predictions.json"
bash "outputs/$run/live.sh" prepare "$pin" "$root" "preflight-$(date -u +%Y%m%dT%H%M%SZ)" "$sha"
BAIGE
```

## Pairing and statistics

The FK evaluator reconstructs the official panel from the eight manifests and 96 original passive tapes. Its full panel must match the digest from the executed FK run. The offline rows join by index, suite, integer task ID, embodiment name and support SHA-256. The committed offline receipt pins the prediction bytes, the seed-41 reader and the ordered support set. Each pair takes both yaw estimates from that input and the same unrounded gain and gripper sign from the historical reader export. The native runner receives one yaw scalar per arm. [Offline implementation and evidence](OFFLINE.md) · [FK evaluator](../symmeta-fk-ur5e-20261123-r1/paired.py)

Both arms run fresh using the verified Goal 6 recovery motor and simulator at `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`, OpenPI at `215abfb217dbac7d5f1273282331b9b1866c0479`, and the LIBERO-Pro2 release at `e5500c690c1498eca1b7050350654964190cdf70`. Both use frozen `pi05_libero`, the native released prompt, the BASE scene, query layout 13, and the original seeded float32 noise bank with one policy proposal per five controls. The original supports use layout 11. The historical prompt, horizon, noise seed and digest, perturbation plan, controller schema and model class are checked per task.

Each shard discards one initial create and runs learned then linear for all 12 transforms. The official callback permits up to three creates before action to match the shared task reset. An executed trajectory has one attempt. The full reset must agree across both arms and all 12 transforms in the new job, including scene physics, observations and frame zero. The opening policy proposal must match within every pair. A pair reaches its final filename through atomic rename after both native outcomes validate. Merge rechecks the exact file set, case identities, shared corrections, reset, opening proposal, native reward count and sum, terminal flags and horizon. [Live evaluator](live.py) · [Launcher](live.sh)

`summary.json` reports **learned minus linear** native success for `all` (8 × 12) and `held_out` (8 × 8, four compositional and four continuous transforms per task). Each panel includes successes by arm, paired wins, losses, ties, the success-rate difference and a 95% percentile interval. The pinned FK statistics function draws eight whole tasks with replacement 2,000 times at seed 20261121, retaining all paired transforms within each task. Per-task counts and differences are included for both panels. Multiply differences and interval endpoints by 100 for percentage points.

## Checked evidence

- **Actual inputs on Baige CPU.** `$root/live-cpu02/` holds the prediction snapshot, panel, SHA-256, config, storage, log and exit receipt. Prepare exited 0 in 21 seconds on code pin `d268f189`. The 96 rows reproduced the FK panel byte-for-byte before yaw substitution (SHA-256 `51a4575ff3105b0e7005befb598f94d8bf69ed0c068324c8833a92b11cfd05d0`). All 96 final rows matched the historical query, tape, gain and gripper fields and their two offline yaw values. The prepared panel SHA-256 is `42b0d845181172270b675e4afd98d99260bd6636c3f687c53695629d5b10040b`.
- **Queued pin and deployment command.** Prepare on `08cf720e` exited 0 in 69 seconds. A field-by-field comparison of the two CPU panels matched all 96 rows and all eight task conditions. The three additional fields record offline revision, receipt digest and source-fit digest. Executing the documented deployment and prepare command produced the same `d268f189` panel hash again in `$root/preflight-20261123T064658Z/` with exit code 0.
- **Runtime.** The four Git checkout checks passed. The policy verification streamed and hashed 16 files totaling 12,439,085,481 bytes. The simulator check matched 88 Python sources and the LIBERO configuration. The shared mount had 257 GiB available and accepted the bundle, checkout and CPU receipts.
- **CPU replay of historical FK records.** A temporary fixture exercised the new shard writer, the official pre-inference callback, and strict merge. Its episode provider returned archived outcomes with their original two correction tuples. Eight warm creates and 192 record calls produced 96 paired files. In archived SymMeta-minus-FK orientation, all 96 gave 5 wins, 7 losses and 84 ties, with success counts 76 and 78. The 64 held-out pairs gave 5 wins, 4 losses and 55 ties, with success counts 53 and 52. Both panels reproduced the original counts and interval after reversing the original contrast. Fresh matched-flow success will be measured by the scheduled native run.
- **Rejection tests on Linux CPU.** Checks rejected an unregistered digest, changed input bytes against the required hash, a duplicate JSON key, missing or duplicate indices, incorrect task or support identity, a non-string embodiment and Boolean or nonfinite yaw. A permutation of the 96 prediction rows reconstructed the same panel. Pair and merge checks rejected a missing arm, changed noise or opening proposal, altered gain, a dropped step reward, an unfinished episode, wrong row identity, a missing pair file and an extra pair file.
