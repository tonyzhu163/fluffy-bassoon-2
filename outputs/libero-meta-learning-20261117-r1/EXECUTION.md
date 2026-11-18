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

The converted official OpenPI checkpoint also finished the same eight episodes
in devqual03 (86250d07). It scored 0/6 distinct cases and 0/2 repeated BASE
controls. Both zero-adapter traces matched exactly and the gradient checks passed.
The first Spatial BASE rotation command changed from -2.055 to -0.004 with
the new motor. Correcting that observed difference was insufficient for success.
This run took 0.078 GPU-hours, bringing the campaign to about 2.56 GPU-hours.

The original JAX reference check uses the same official checkpoint, the same
two BASE resets, the native OpenPI sampler and five executed controls per query.
It verifies the initial simulator state and public observations against
devqual03. Changes in sampler, implementation and replanning cadence are
joint here. These are development checks, with all eight official test tasks
reserved and the training collection still pending.

On Nov 18 the original JAX policy (devref01, 9d4a4635) solved both matched
BASE cases, Spatial in 106 controls and Goal in 87. Simulator reset hashes,
public inputs and the first query noise agreed with the converted policy's
panel. The JAX run used the OpenPI defaults of ten flow steps and five executed
controls per replan. This establishes a competent nominal source on these two
resets. Compare the learner-compatible PyTorch policy at those same settings.
The first retry (devqual04) stopped in the GPU probe during handover.

Use the original JAX policy to collect the registered 96 fitting episodes.
Keep both camera inputs, state, noise, proposed and issued actions, issuance
masks, native rewards and boundaries. Compute learner features from these
saved public inputs after fixing the learner-compatible policy. Use the same
BASE instruction in both timing conditions and preserve the released prompt
as metadata. This pilot measures corpus coverage and outcome diversity.

Devqual05 (6ddd4381) completed the same six distinct cases and two repeat
controls at the reference cadence. All failed. A CPU replay of the first
Spatial query with the official paired weights/statistics found identical
images, image masks, token IDs and token masks. With identical public state
and a common synthetic action vector, the two transform pipelines differed
by up to 1.183 in normalized state and 0.623 in decoded action. The pinned
RLinf runtime used mean/standard-deviation transforms. The released pi05_libero
policy uses quantiles. Apply quantiles to both transforms of the converted
official checkpoint and repeat the same paired development panel.

Nov 19, 00:03 Shanghai. After enabling quantiles, the replay matched the
original JAX transforms exactly for the normalized public state, decoded
test actions, three camera slots and their masks, tokens and token mask.
Devqual06 was submitted on the shared development 4090 to check whether
this numerical fix restores closed-loop success.

The original JAX fitting pilot devcollect01 (787ada19) finished 96/96 episodes.
There were six attempts in each cell below, grouped into two chronological
three-attempt trials. Each BASE/S20 pair shared reset IDs, noise seeds and
instruction. Counts are native successes.

| Fitting suite and task ID | BASE | S20 delay |
| --- | ---: | ---: |
| LIBERO-10 2 | 6 | 0 |
| LIBERO-10 9 | 6 | 0 |
| Goal 5 | 6 | 0 |
| Goal 7 | 6 | 3 |
| Object 2 | 6 | 0 |
| Object 6 | 6 | 0 |
| Spatial 2 | 6 | 0 |
| Spatial 4 | 5 | 0 |
| Total | 47/48 | 3/48 |

All 96 archives were read back and checked against their case, control count,
issuance mask, terminal marker and reward. The panel contains 4,628 policy
queries and 23,024 executed controls in 496.5 MiB of compressed arrays.
Issued actions are stored as float32 and match the float32-cast proposals
exactly. The masked float64 proposals preserve the original commands. The
largest cast difference was below 6e-8. Cumulative cost through this pilot
was 3.18 of 200 GPU-hours, including the RoboCasa bridge. All delayed
successes came from Goal 7. The seven other fitting tasks need successful
delayed examples or teacher actions before this corpus can support a broad
comparison of how learners recover from delay.
