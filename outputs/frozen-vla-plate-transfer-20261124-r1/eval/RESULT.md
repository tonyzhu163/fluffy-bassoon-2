# Both frozen adapters failed on the first 64 Kinova3 plate starts

The registered comparison stopped for harm at its first look. On 64 independent, freshly generated Goal 5 placements, published π₀.₅ succeeded 62 times. Both the Panda-only and mixed Panda/Sawyer adapters failed on all 64. Mixed lost 62 stock successes and rescued none. Two placements defeated all three policies. [Paired native returns](evidence/look-64.json) · [Independent endpoint and statistical analysis](../stats/evidence/look-64.json)

| Frozen condition | Native successes | Success rate |
| --- | ---: | ---: |
| Published stock π₀.₅ | 62 / 64 | 96.875% |
| Panda-only history adapter | 0 / 64 | 0% |
| Mixed Panda/Sawyer history adapter | 0 / 64 | 0% |

Mixed minus stock was −96.875 percentage points, with a sequentially valid simultaneous 95% interval of −99.94 to −64.32 points. The two-sided exact McNemar p-value was 4.34 × 10⁻¹⁹ at the registered first-look threshold of 0.005. The closed secondary gate leaves the question of a source-diversity benefit unresolved. The observed mixed-minus-Panda-only difference was zero. The remaining 192 placements in the sealed 256-case manifest were never scored. [Stopping rule](../stats/PLAN.md) · [Statistical readback](../stats/evidence/look-64.json)

## Physical starts and deployment

Sampling used the prior native BASE placement generator, independent indexed placement and diffusion-noise streams, and the first valid draw for each case. All 256 placements were frozen before scoring. The release rejected one candidate. The admitted placements were distinct from all 253 eligible training-pool geometries on all four comparisons of seed, donor state, full scene and movable XY coordinates. The manifest preserves the physical parameters, original donor state, draws, rejections and shard for each case. [Native enrollment](evidence/manifest.json) · [Historical placement inventory](evidence/prior.json) · [Source overlap readback](evidence/pre-score-source-overlap.json)

Each case ran all three policies on one RTX 4090 from identical native reset physics, measured joints, camera observations, diffusion noise and decoded opening commands. The reader consumes actual signed finger positions and public tool motion from executed controls. Each policy replanned every five controls through native terminal success or the full 300-control horizon with relative OSC on MountedKinova3. Independent readback reconstructed both cameras and every policy input, checked the causal history and issued commands, and verified physical and observed equality while the controls coincided. [Full trajectory index and per-case SHA-256 receipts](evidence/readback-64.json)

Genuine stock ran its published sampler at every replan. Both fitted policies used the same post-opening compiled program on each GPU. On each mixed trajectory at control 5, unissued fixed-input readbacks compared the learned offset, zero offset and genuine stock. Across the 64 cases, the median maximum absolute decoded coordinate difference over a ten-action chunk was 0.00161 for zero addition against genuine stock and 0.286 for the trained mixed offset against zero addition. These measurements quantify proposal changes at a fixed observation. The experiment measured terminal success for the three deployed policies. [Fixed-input diagnostics and weight identities](evidence/accounting-64.json)

## Execution and evidence

The first four-card request had zero allocated runtime. The parent submitted four independent one-card Baige Distributed Training jobs with the identical frozen code and fit. Shard 0 was job-7ox4z7a84yd6, shard 1 job-jtk91dats32z, shard 2 job-ft1k3i7ds5zy and shard 3 job-6igzxeqdhhuz. Each completed 16 triplets and exited 0. Together they logged 192 episodes, 45,283 native controls and 4,128 launcher GPU-seconds (1.147 GPU-hours). Median wall time per case was 42.78 seconds. The full two-camera HDF5 trajectories, physical states, every reward, proposals and execution receipts occupy the four shard roots on Baige PFS under `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-plate-transfer-20261124-r1/eval/gpu01s{0,1,2,3}`. The index records each exact path. [Submissions](evidence/submitted-64.txt) · [Native tape index](evidence/readback-64.json) · [Compute and archive accounting](evidence/accounting-64.json)

Evaluation code ran at `796acc7fa4ff8fb536eaa705a4e9d83c86cda0c3` with fit code `7f749ae21b8d74ad58536c00ae4aafd5ffca45c5`. The fitted-family SHA-256 was `7d845ce26bbb7f7f5da45b4885660023dd0e33b622cb9913547d50290ca16fe2`. All four shards recorded identical stock, deployed-copy, adapter, normalization and fit-file hashes before and after rollout. The final 64-case seal SHA-256 was `c37059501e6fe4eaffba5b9e8d4ebdb8ca5f07d6e5f85b0e481e562022aa134d`. [Frozen weight accounting](evidence/accounting-64.json) · [Native CPU qualification](evidence/cpu-fixture.json) · [Durable storage qualification](evidence/storage-fixture.json)
