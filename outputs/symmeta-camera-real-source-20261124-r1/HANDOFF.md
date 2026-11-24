# Source camera exposure follow-up

The render job writes one verified nominal/moved image sequence per locked source ID. Cache and fit read previous stages via immutable directories. The cache requires the completed render exit receipt and the prior source bank at prep01/bank and offset01/bank. Its wall allowance subtracts all previous CPU attempts, including the unsuccessful diagnostic. Both Linux stages record CPU idle counters and disk capacity. The fit validates the content hashes of the cached arrays, the disjoint split, and the exclusive 4090 runtime.

From the parent-deployed worktree, run the following commands with a full matching revision at each stage. Source rendering and feature preparation use only CPU. Parent owns submission to Distributed Training and passes the real digest from the ready receipt. Select 'both' after the preregistered independent Kinova3 scorer releases its primary readback. Until then 'ur5e' runs the registered fit and UR5e comparison without holding a GPU for Kinova3.

```sh
p=outputs/symmeta-camera-real-source-20261124-r1
bash "$p/prepare.sh" "$render_pin" render02
bash "$p/cache.sh" "$final_pin" render02 cache01
bash "$p/gpu.sh" fit-score "$final_pin" fit01 cache01 "$ready_sha256" both
```

The separate release command if needed is `bash "$p/gpu.sh" score-kinova "$final_pin" kinova01 fit01 "$seal_sha256"`. Keep aggregate allocated GPU time within the agreed 1.3 hours. The forecast is about twice the prior pair's 384 seconds plus validation, startup and export. A single job runs the same four freshly trained models and both sealed target banks. The job exits as soon as its results have been written.

Portable outputs include exact case IDs, digital IDs, the measured geometry and missingness, monitor curves and all factor predictions, masks/metadata controls, paired task/seed reductions, exposure and checkpoint digests. Native inference export is the selected-budget rows with assigned=true and calibration=true. The parent decides on an independent native rollout using the two preregistered case files. The source fit never reads them.
