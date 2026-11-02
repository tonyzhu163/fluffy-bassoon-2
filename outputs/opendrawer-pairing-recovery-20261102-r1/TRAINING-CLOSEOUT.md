# Training allocation closed after all four plateaus

AD/full and AD/static converged at1900, DPT/full at23800 and DPT/static at24100. FULL-RECOVERY-20261102-1446.json in the original training output records each latest checkpoint hash, nonempty AdamW state, sampler/Python/NumPy/Torch/CUDA RNG keys and converged plateau. On allocated Linux CPU the full payloads were loaded and verified; their selected/latest hashes were rechecked at closeout.

Only owned Baige job-8kb761dxfvtm was stopped, after confirming all recoveries and exact job/source. Console terminal state is stopped at Nov2 14:48:58 Shanghai (pod14:48:54), created11:48:26,2 GPUs. Conservative creation-to-terminal allocation charge is21,664 GPU-seconds. This is a terminal accounting value, not a scientific resource cap. training-stopped.png records the terminal console.

The immutable old driver began legacy evaluation before this closure:30 complete episodes,23,128 conservative controls across2 journals,2 unresolved calls fully charged. These rows lack the repaired native snapshots and are excluded from the new comparison. Preserve all records and checkpoints; no training is resumed. ADMISSION-20261102-1500.json contains exact journal hashes and charges. Complete source/PPO collection costs and diagnostic/readout attempts remain required in the final resource accounting.

The repaired AD and DPT/full phases have4 passing2-GPU manifests each,16 globally distinct scheduler UUIDs,64 and67 complete episodes at this snapshot, and no worker tracebacks. The first6 AD and3 DPT/full endpoints independently pass retained snapshot/canonical checks and terminal simulator feedback verification. These are partial-readout admission checks, not success-rate results. DPT/static job-5qtsrh7a1khb is separately queued.
