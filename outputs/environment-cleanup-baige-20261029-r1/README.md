# Baige environment cleanup

Completed 2026-10-29 at 18:17 Asia/Shanghai. Scope: the environments under
`/mowenjie/fluffy-bassoon-2/envs`; the original repository's shared runtimes
remain in place.

| Location | Before | After |
| --- | --- | --- |
| Current project | 5 virtual environments + 2 package overlays | 1 virtual environment + 2 package overlays |
| Original project, `/mowenjie/fluffy-bassoon/envs` | 7 virtual environments + 1 package overlay | Unchanged |

Retained in the current project: `metaworld-3.1.1-r1`,
`dsrl-pi0-overlay-20261029-r1`, and `pytest-overlay-20261029-r1`.
The Meta-World environment and RLinf loader are available for the new screen;
the overlays support the existing LIBERO pipeline.

Archived and removed the expanded copies of:

- `dsrl-can-20261029-r1`
- `dsrl-can-20261029-r2`
- `metaworld-baselines-20261028-r1`
- `robotwin-baseline-smoke-20261028-r1`

The Can comparison, ML10 baseline campaign and RoboTwin gate are closed in
their audited run records. Both Can environments were retired together:
r2 loads packages from r1. The retained Meta-World environment also supplies
the retired Meta-World baseline environment. Dependency checks found no
retained environment loading a retired one. Checkpoints, inputs, Python
installations, shared environments and the uv cache were preserved.

All five current-project virtual environments were created with uv. Package
inventories were captured with `/mowenjie/bin/uv` (0.13.1), together with the
local distribution versions, `pyvenv.cfg` and `.pth` dependency paths in
[inventory-before.json](inventory-before.json). Some historical environments
borrow packages from other runtimes; their full archived contents and those
dependency paths are required for faithful restoration.

The recovery archive is on Baige at
`/mowenjie/fluffy-bassoon-2/outputs/environment-cleanup-baige-20261029-r1/retired-envs.tar.zst`.
It contains 2,839,859,200 uncompressed tar bytes in 859,168,534 compressed
bytes (about 819 MiB). Zstandard integrity and `tar --compare` against every
source member passed before removal. Its SHA-256 is recorded in
[archive.sha256](archive.sha256). These sizes do not measure reclaimed
filesystem space: uv may share hard-linked files with its cache.

[cleanup-complete.json](cleanup-complete.json) records the verified final
counts and archive hash. Run [restore.sh](restore.sh) on the Baige development
endpoint to restore all four environments at their original paths; it
refuses to overwrite any existing environment.
