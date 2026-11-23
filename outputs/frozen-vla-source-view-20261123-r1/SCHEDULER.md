# Source-render native evaluation

Baige Distributed Training accepted job `job-69hluhi5wgb7` at 19:12 Shanghai on November 23. The allocation requests one RTX 4090, 12 CPU cores and 48 GiB for up to 60 minutes, with zero automatic retries. It was waiting for an allocation at 19:13. [Baige receipt](scheduler.jpg)

The deployed Git revision is `2b59cff7e173ee8064905256b98a1dd3fd0f79d5`. The launcher writes the GPU probe and paired policy outputs to `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-source-view-20261123-r1/gpu01`. The readback worker verifies matched policy inputs, repeated proposals, native rewards and any invalid rendered poses. [Input checks and reproduction command](HANDOFF.md)
