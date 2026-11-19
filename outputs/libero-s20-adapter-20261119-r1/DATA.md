# Released demonstration inputs

The source is LIBERO-Pro2.0-Train at `6c4c5524676815a88880a7025badd17fb724f7f4`.
The sampler takes BASE and S20 from the 24 registered fitting tasks and filters
copies declared by the publisher. It verifies the cached publisher indexes and
the byte length and SHA-256 of each of the 48 HDF5 files. Downloaded files live
in the Baige NVMe setup directory, with the four existing bowl archives reused.
The official host timed out from Baige, so the CPU staging job used the public
HF mirror for this public data. The source pin and file hashes remain the same.

The sampler balances BASE and S20, then samples a task, reset, trajectory and
observation in that order. It groups each task/reset identity across cases for
the train and validation split and retains training coverage of each case on
each task. Each observation is followed by up to ten issued-action targets,
beginning at action `t + 1` for observation `t`. A mask excludes unavailable
actions in the one retained short S20 trajectory. Both timing arms use the
archived BASE instruction for the task. The loader returns raw archive camera
arrays, which use the existing live `reference.observation` transforms.

The bowl smoke index contains 120 trajectories in 67 training and nine
validation blocks. The OpenPI interpreter loaded 24 sampled windows and
reproduced every raw field and following-action target from the pinned HDF5.
The sampled training and validation batches each contained six BASE and six
S20 windows. The archived task instructions already agreed across timings for
these two tasks. The normalized archive camera orientation matches the existing
policy input after the two-axis flip in `reference.observation`.

Baige artifacts are under
`/mowenjie/fluffy-bassoon-2/outputs/libero-s20-adapter-20261119-r1/data/`.
They include `index-smoke.json`, `smoke-check.json`, the per-archive download
progress and the completed fitting index `index.json`.
