# Baige CPU execution

The synthetic self-test and both archived-tape analyses completed at the owner's verified pin `b1eb9dd72be6cff3dcedfb256ecd47e176dc25e6`. All three exited 0 in **0.635676 seconds** on the existing CPU allocation. The two per-case reports are available locally. [Physics-based estimates](evidence/cpu01/estimate/report.json) · [Factor-supervised estimates](evidence/cpu01/factor/report.json) · [Invocation and timing receipt](evidence/cpu01/execution.json)

## Allocation and inputs

At 15:32:23 Shanghai time on November 20, 2026, the existing JupyterLab terminal on `mwj-dangerous-dogs` (`dv-63739b1202`, host `aibox-r855e239307b-6bfb688cdd-lz2dx`) completed the preflight. The Linux cgroup allowed eight CPU cores and 32 GiB RAM. The CPU allocation exposed zero NVIDIA card device nodes.

The project volume at `/mowenjie` reported GPFS, 418 GiB available out of 2.0 TiB and 1,721,115 free inodes. A temporary file written in the new run's `setup/` directory read back correctly. The container lacked both `quota` and `mmlsquota`, so its per-project storage limit could not be read. The existing project interpreter at `/mowenjie/fluffy-bassoon-2/setups/libero-meta-learning-20261117-r1/env/bin/python` loaded Python 3.11.14 and NumPy 1.26.4.

The H1 checkout at `/mowenjie/fluffy-bassoon-2/setups/embodiment-command-frame-20261120-r1/source-d2c7d4e9` reported `d2c7d4e9e9310ccdc1abc11109c4ad64c30d8d13`. Input `/mowenjie/fluffy-bassoon-2/outputs/embodiment-command-frame-20261120-r1/native02/collection/tapes.jsonl` contained 256 lines, 2,831,993 bytes and matched SHA-256 `c1207da02bd988de7fb51cab00a894ed29f1a31ca7d7306bfe9c9c7fb05c3ebf`. The registered analysis base is `1967f2b31fd4f64c0c9771fb04df2a268c1e2e41`. [Plan](PLAN.md)

## Deployment and execution

The Linux mirror had the H1 pin and lacked the analysis base. A 257,110-byte incremental Git bundle, SHA-256 `f0bd3bdb6280cea9f40eddcadd04060bae36697ca05e218b018de8ad98abe6bd`, verified on both machines. The new sparse checkout at `/mowenjie/fluffy-bassoon-2/outputs/embodiment-tape-observability-20261120-r1/setup/source-b1eb9dd7` had a clean, detached HEAD. The plan and both estimators matched their Git objects. The SHA-256 values for `estimate.py` and `factor.py` were `4ee95d7ebe62415699d0da755ff613a01fbf46f943f8c29604df4bfbdfd88eef` and `79a084f75928219ec4c995af8c6ed4aecbe5471193e244f75b2efa739f2d7dcd`.

The timed sequence started at 15:39:42.801 Shanghai time and finished at 15:39:43.437. OpenBLAS, MKL and OpenMP each had a one-thread limit. A shell quoting error in the timing wrapper was corrected before the output directory was created or any analysis stage ran. The measured execution used fresh sibling directories under `/mowenjie/fluffy-bassoon-2/outputs/embodiment-tape-observability-20261120-r1/cpu01`. The run used 0 GPU-hours and the existing source archive supplied every observation. [Exact commands](evidence/cpu01/execution.json)

| Stage | Elapsed seconds | Exit | Receipt |
| --- | ---: | ---: | --- |
| Rotation and lag self-test | 0.247714 | 0 | [Self-test log](evidence/cpu01/self-test.log) |
| Physics-based estimator | 0.201253 | 0 | [32 cases per response model](evidence/cpu01/estimate/report.json) |
| Factor-supervised reader | 0.184642 | 0 | [32 cases per reader](evidence/cpu01/factor/report.json) |

## Evidence readback

The 45,277-byte return archive matched the Linux SHA-256 `b5d8e035be04c6bbc2ec37d7648c121c0d39b3d916b15378574ec9f4a3ebcfea`. All six payload hashes matched locally. The physics report SHA-256 is `b6acf89b4e5f245391e886d5a0ebb01351d24687737cd508b2171417fe501c59`. The factor report SHA-256 is `8dcd9a2fb1eb3e005c82744e21fc4bd5f9b13aebf7092e9bbe2783ffc321152c`. The raw JSONL stayed on Baige. [File manifest](evidence/cpu01/SHA256SUMS)

For each of two response models and three supervised or constant readouts, local checks recovered the complete set of two arms × two settings × eight layout indices. The two donors reversed correctly within every matched pair. Recomputing scores from per-case JSON reproduced the reported physics-model mean losses and nearest-angle counts, the factor-reader mean and median errors, and signed paired separations.

| Arm and held-out mapping | Four-tap nearest yaw | Matching query MSE | Swapped query MSE | Fixed common query MSE | Four-lag supervised nearest yaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Panda, +45°, gain 1 | 7/8 | 0.73167 | 0.90240 | 0.84689 | 8/8 |
| Panda, +90°, gain 0.65 | 7/8 | 0.70038 | 0.88240 | 0.77021 | 2/8 |
| Sawyer, +45°, gain 1 | 7/8 | 1.19404 | 1.30808 | 1.19324 | 8/8 |
| Sawyer, +90°, gain 0.65 | 6/8 | 1.17722 | 1.30227 | 1.16886 | 2/8 |

The four-tap estimator identified the nearer yaw in 27/32 cases, compared with 20/32 for one tap. Matching practice reduced normalized XY query MSE from 1.09879 to 0.95083 against a swapped donor. All eight layout-index averages favored matching practice. The paired-bootstrap 95% interval for the reduction was 0.12171 to 0.17925. Against the fixed common predictor, the interval was −0.06063 to 0.14074. Mean signed donor separation was 55.30° on Panda and 10.06° on Sawyer for a true 45° gap. The four-lag supervised reader scored 20/32, the zero-lag reader 21/32 and the constant 16/32. The familiar-arm tapes contain usable mapping information, with limited calibration on this previously examined panel. [Physics cases and layout summaries](evidence/cpu01/estimate/report.json) · [Factor-reader cases](evidence/cpu01/factor/report.json)
