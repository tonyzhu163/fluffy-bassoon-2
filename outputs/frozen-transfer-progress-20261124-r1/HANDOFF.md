# Parent launch handoff

The parent deployed the revised verifier and registered controller at the following revision. Use this same revision for qualification and both native shards.

```bash
CHECKOUT=/mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-d70e206a00c6
PIN=d70e206a00c6c5d59f190ec4a9b0bc6df2ec82c4
D="$CHECKOUT/outputs/frozen-transfer-progress-20261124-r1"
```

The runtime imports the frozen motor and simulator from the native transfer checkout at a8a997e7d06bb131d3f9710204c58e106f0f0904. The executable checks the release revision, OpenPI revision, first-completed PR7 prediction bytes, source response coefficients, support hashes and model weights.

## CPU qualification

**Passed on Baige.** The cpu02 attempt exited 0 on cores 44 and 45. It completed 32 scripted episodes and 352 native controls per arm, 704 in total. The released-row fingerprint matched the original bank in all 16 arm/task cases. Both arms also matched on the eight exogenous task and noise schedules. The progress controller selected a different donor index from fixed at 15 of 16 eligible IIWA boundaries and 16 of 16 Kinova3 boundaries. On those scripted trajectories it changed 43 and 48 actual sent commands respectively compared with the fixed choice on the same history. These are wiring measurements. The native π0.5 success contrast awaits the two scheduler jobs. [IIWA receipt](evidence/IIWA-cpu02.json) · [Kinova3 receipt](evidence/Kinova3-cpu02.json) · [Release banks](evidence/release-banks.json) · [Independent readback](evidence/cpu02-readback.json)

The first CPU attempt caught a verifier error on the object suite. Its official arms use OnTheGroundIIWA and OnTheGroundKinova3. The second attempt used the matching per-suite classes and reran all 64 episodes. All 16 files of the frozen checkpoint passed byte verification on Baige. A separate reduction check rejected changed donor indices, wrist and gripper commands, noise, and measured displacement. [First attempt](evidence/cpu01-stop.json) · [Checkpoint](evidence/weights.json) · [Reduction fixtures](evidence/reducer-qualification.json)

The command below records the completed qualification. The fixture exercises the normal native run_episode path, real camera inputs and physical first steps for all four modes, both arms and all eight tasks, with no model inference. It uses a scripted H=13 proposal and 11 controls to exercise a final partial block. Separate numerical fixtures cover H=9 rejection, H=10 acceptance, all six nearest-prefix indices, ties, saturation and gain/yaw, absolute-origin invariance, an unchanged first proposal and exclusion of future suffix values. The CPU endpoint is qualified.json for each arm under the new attempt. Read exit.txt for the combined exit code.

```bash
PHASE_CPUS=44,45 bash "$D/qualify.sh" "$PIN" cpu02
```

## Two native job commands

Use two independent single-card Baige **分布式训练** allocations, an RTX 4090 or RTX PRO 6000D for each, the native LIBERO image and shared /mowenjie mount. Set the scheduler wall limit to one hour per allocation. The launcher allows 3,540 seconds plus five seconds' kill grace. A new attempt name prevents overwrite. GPU probing and device-qualified NVIDIA EGL occur inside the job. Model loading and all native policy computation take place there. The parent owns submission and monitoring.

```bash
bash "$CHECKOUT/outputs/frozen-transfer-progress-20261124-r1/launch.sh" --fresh "$PIN" IIWA gpu01 cpu02
bash "$CHECKOUT/outputs/frozen-transfer-progress-20261124-r1/launch.sh" --fresh "$PIN" Kinova3 gpu01 cpu02
```

Both write under /mowenjie/fluffy-bassoon-2/outputs/frozen-transfer-progress-20261124-r1. The two arm folders are gpu01-IIWA and gpu01-Kinova3. Each contains gpu.json, origin.json, run.log, an episodes directory with one atomic JSON per fresh native endpoint, exit.txt and result.json after 32 validated episodes. Check the first complete quartet and advancing episode count, then at 20-minute intervals. A missing episode leaves the arm incomplete. Preserve partial receipts and diagnose before any new charged allocation.

## Independent reduction

The analysis requires Python with NumPy, the deployed study checkout and the two shard directories. It rebuilds all 16 matched keys, replays selection, canonical clipping/blending, inverse correction and exact sent actions from the retained original proposals, reconstructs the independent noise bank, and validates native terminal flags and reward tapes. It reports native success for all four modes, paired changes, task-cluster intervals and exposure. It requires every registered episode for pooled results.

```bash
R=/mowenjie/fluffy-bassoon-2/outputs/frozen-transfer-progress-20261124-r1
/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B "$D/readback.py" \
    --iiwa "$R/gpu01-IIWA" --kinova3 "$R/gpu01-Kinova3" --out "$R/paired-layout31.json"
```

The original source model measures five-control aggregate displacement. Its sub-block interpolation is fixed analytically. The archived passive tapes retain three valid blocks and were collected with a 15-control cap. The older NPZ format omits individual block lengths. The replay on Kinova3 goal/3 layout 31 used fixed tapes in an earlier drawer study. This design addresses a fresh native policy comparison on those released starts. [Design](PLAN.md)
