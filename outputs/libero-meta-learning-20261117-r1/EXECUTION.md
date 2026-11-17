# Execution

November 17, 2026, Shanghai. The user approved proceeding with the LIBERO
meta-learning design. This continues the existing200 GPU-hour campaign;
the completed RoboCasa bridge charge is2.36 GPU-hours. No LIBERO GPU job has
been submitted yet.

Main reviewed and pulled through dad41343 before implementation. The first
setup source is97dd915d. LIBERO-Pro2.0 is pinned to
e5500c690c1498eca1b7050350654964190cdf70, transferred as a Git bundle and
checked out on Baige. This is the current Pro2.0 release, not the pre-existing
older LIBERO-PRO asset set. Shared PFS had approximately83 GiB available;
the setup performs an actual write probe. The quota utility is unavailable.

An isolated runtime inherits the qualified frozen-policy dependency set and
installs its own robosuite1.4.0/MuJoCo3.2.3 simulator packages. It does not
modify another environment. Optional vendor assets total175,877,791 bytes
in76 Git LFS objects; payloads are individually SHA-checked and stored in a
separate setup tree. Source-relative asset resolution is handled by a byte-
identical copy of simulator Python files in the setup tree, with the hydrated
assets adjacent. The upstream Git checkout remains unchanged; the setup
receipt hashes each copied Python file. No complete new demo dataset is downloaded.

Candidate motor: existing RLinf/RLinf-Pi05-LIBERO-SFT checkpoint revision
45ccfcc4e28634f1576ebf78cab0fbe2fd82432d. Its preserved acquisition receipt
records model SHA256
4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7
and normalization SHA256
dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1.
Fresh hashes matched both preserved digests on November 17 before execution. This is a different release
from the earlier audited OpenPI/LeRobot pi05_libero weights; equivalence is
not assumed. The candidate is a LIBERO-SFT checkpoint, and no evidence here
establishes that the backbone excluded our eight meta-test tasks. Report
meta-training task holdout only.

The qualification uses two development tasks declared in SPLITS.json:
libero_spatial1 and libero_goal1. Each worker runs native BASE, paired
zero-adapter BASE, camera caseS15 and timing caseS20: eight episodes total,
including failures. The first native query also checks exact zero-adapter
action parity, three flow-gradient updates and frozen motor parameters.
These checks use native action targets for numerical wiring only; they are
not teacher competence, trained adaptation or history-benefit results.

Proposed allocation: two one-GPU Pro6K workers,45-minute worker deadlines,
50-minute scheduler maximum, automatic retries off. Reserve at most
1.667 GPU-hours including startup; diagnose before any retry with a fresh
output identity. Queue and actual GPU probes remain required before launch.
The model predicts/executes10 controls per query with five denoising steps,
matching the RLinf Pi0.5 template; LIBERO-specific image rotation,8D state,
7D actions and normalization replace the RoboCasa interface. The selected
checkpoint's input transform is stateless for the motor; public proprioception
remains available to the history learner. Confirm the native configuration
and behavior before treating the runtime as qualified.

The user clarified that the history corpus must be large enough. PPO is not
mandatory for every history: frozen SFT behavior supplies genuine attempts,
and teacher queries supply DPT labels where a competent teacher is available.
DIT/IC-IQL need informative action/outcome variation, while an AD claim about
distilling improvement needs actual improving source streams. Do not relabel
unordered demonstrations as PPO learning histories.24 fitting task IDs are
the pilot coverage, not evidence of sufficient scale; also measure semantic
skill diversity, independent task-condition trials, unique query coverage,
success/failure coverage and learning curves before choosing bulk volume.
Prefer one shared corpus and selective PPO source improvement where needed.

The first direct CPU dependency installation stalled before completion and
was stopped after verifying its exact owned PIDs. No GPU was allocated.
Official PyPI wheels for robosuite, MuJoCo, NumPy and Pydantic were downloaded
and checked against PyPI SHA256 metadata, then transferred for local install.
A CPU preflight checks imports, simulator source copies, the published BDDL
manifest, image orientation and public state packing. GPU and simulator
qualification still require the actual scheduled job.

The corpus plan now includes nested 8/16/24-task development curves and
independent trial-depth curves, with semantic coverage reported separately.
No bulk episode count is committed before measured costs. Commands issued
to env.step are the public action history; environment-internal delayed or
filtered controls are excluded from student inputs.
