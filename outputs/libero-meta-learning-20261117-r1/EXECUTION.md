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

November 18, 09:36 Shanghai: resumed after interruption. The isolated
simulator package install completed on November 17, but asset setup and
CPU/native qualification are not confirmed complete. No LIBERO GPU job was
submitted before interruption. The development SSH endpoint now times out;
console browser control is unavailable in this session. Requested restored
Baige access. No additional GPU use is known; the closed charge remains
2.36 GPU-hours and the total cap remains 200.

The exact nested 8/16/24 fitting-task subsets are now fixed in SPLITS.json.
Local recovery preserved 67 of 76 optional vendor assets; the other nine
downloads were too slow and were stopped, with partial files excluded.
These vendor replacements are not required by the planned BASE/S15/S20
qualification. Once access returns, check any surviving setup process,
finish the core setup and CPU preflight, refresh queue capacity, then submit
the prepared two-worker qualification with a fresh verified main pin.

November 18. Baige SSH recovered. The allocated development RTX 4090 ran
the two fixed cases in sequence. Attempt devqual01 stopped loading the
policy because its transform module imports RoboCasa. Separating the policy
and simulator interpreters resolved the dependency conflict. A native reset
and step passed. Attempt devqual02 (debd2bc0) completed eight episodes.
Both zero-adapter replays matched every issued action. Both gradient checks
passed. The RLinf SFT motor failed all six distinct task/condition cases,
including both nominal settings. The other two episodes are repeated nominal
controls. On Spatial BASE, the first rotation coordinate stayed near -2.05
across 220 controls. No learner comparison has been made.

For the next qualification use the already available official OpenPI pi05_libero
weights and their bundled action statistics. Convert the Orbax weights using
the pinned OpenPI converter (ca4c6d71, bfloat16, CPU), then run the same
development panel. This changes the candidate motor as a unit. Its causal
contribution cannot be assigned to normalization alone. The pilot collector
is prepared for 8 fitting tasks x 2 conditions x 2 trials x 3 attempts
(96 episodes), after the baseline supports meaningful collection. The two
repeats use different initial states and matched draws across conditions.
