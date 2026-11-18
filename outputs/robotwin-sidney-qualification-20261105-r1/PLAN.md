# Native Sidney pi0.5 qualification

User selected Sidney on 2026-11-05 as the next SFT initializer to try before
RoboTwin PPO. This is a fresh run; the queued Motus qualification was stopped
before allocation and consumed zero GPU-hours. No success comparison exists yet.

Model: `SidneyXie/pi05_robotwin`, immutable revision
`e49e2ab6c11f07511573b67261bd129e88d0a416`. Full file identities are in
`HF-MANIFEST.json`; saved configs and model card are in `checkpoint/`.
Keep native H50, K10, three RGB cameras, 14 absolute joint targets, and saved
mean/std processors. No Motus quantile, joint calibration, or delta transform.
Disable compilation for the initial correctness screen and record that runtime
choice. Strictly verify all checkpoint tensors are loaded; the historical
LeRobot loader can catch a loading error and return an uninitialized model.

First qualify one real source-task episode on allocated Baige hardware, including
GPU probe and renderer PCI binding. Then screen source development cases, first
the three tasks with existing scene banks. Qualify new banks for the remaining
three before claiming a six-task screen. No held-out task selects the initializer.
Success rates from the model card are external claims, not local measurements.

Use isolated model and simulator Python processes if required by dependency
versions, retaining the existing SAPIEN runtime. Pin and record both source and
runtime revisions. The interface carries only RGB, joint state, instruction,
and absolute action arrays. Native inference must pass before any explicit
RLinf format conversion, K5 migration, numerical replay check, or PPO update.

Initial allocation budget: at most 4 GPU-hours including qualification failures,
with wall-time limits. User-approved overall ceiling is the repository's
50 GPU-hour incremental scope; record cumulative usage. No SGD in this run.
Deploy committed scripts via Git to a fresh detached Baige worktree and record
the exact commit before launch. Each attempt gets a new output directory.

Camera profile for the native screen: the pinned RLinf RoboTwin preset uses
D435 at320x240. Use the existing Large_D435 head/wrist presets at640x480,
with the same37-degree field of view, matching Sidney's saved input metadata.
This camera setting is explicit; do not label it the unmodified RLinf preset.
Verify all twelve tasks' object-family files before another GPU allocation.
An official-archive restore performs file I/O only and preserves matching
existing assets; it does not overwrite conflicting shared files.
