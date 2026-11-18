# Initial Sidney performance and RoboTwin recipe diagnosis

The user requested an initial-policy performance test before checking the RLinf PPO/GRPO recipes. This is a narrow phone-stand diagnostic using the untouched broad Sidney SFT checkpoint. It does not rerun the 32-task published benchmark, switch to task-specific SFT, take new optimizer steps or alter either completed matched arm.

Use the first 16 entries of the pinned existing phone-stand evaluation bank, without outcome filtering or replacement. They are disjoint from its TRAIN bank. The pinned RoboTwin benchmark limit for this task is already 400 controls; all conditions retain H400, absolute 14D actions, H50, Large_D435 head and wrist views, the saved mean/std processors, and matched seeded initial Gaussian draws. Batch size is one in every condition.

| Condition | Policy implementation and inference | Observation path |
| --- | --- | --- |
| native10 | Original strict-loaded LeRobot Sidney, ten ODE steps | Native full observations |
| port10 | Qualified RLinf Sidney port, ten ODE steps | Native full observations |
| port5 | Same RLinf port, five ODE steps | Native full observations |
| sde5_full | Same untouched port, five-step Flow-SDE, noise0.3 | Native full observations |
| sde5_fast | Identical Flow-SDE policy | Released RLinf fast path used by the completed training pair |

Rotate condition order by case. Verify actual seeds, scene information, instructions, public RGB/state hashes and exposed articulation state before actions. Persist failed comparisons before rejecting pairing. Keep model RNG isolated from simulator RNG. Every model remains frozen; converted parameter versions and the native worker's strict loading/frozen checks are retained. The native worker and model runtimes are the already qualified immutable sources; no library upgrade is needed.

Full-observation conditions retain measured executed-action masks. The fast path continues to report requested/conservative controls explicitly. No full native physics snapshot, bit-exact closed-loop replay, success-rate reproduction or student data admission is claimed. Sixteen cases are a diagnostic sample, not a precise benchmark estimate.

Allocate one Baige Pro6K GPU,16CPU,128GiB RAM,16GiB shared memory,240-minute maximum, automatic retry OFF. Mandatory allocated-step GPU/renderer checks, actual simulator episodes, host-memory logging, storage/quota/write checks and a 12GiB output cap apply. Prior conservative cumulative cost is130,567GPU-seconds; this allocation reserves14,460 including launch margin, for145,027 (40.28528GPU-hours) worst case below the existing50-hour cap. Other owners/floor are conservatively reserved at286GiB. No active cache, checkpoint or completed source output is moved.

After the initial-policy result, audit both the pinned and current public RLinf RoboTwin recipes: identify which models/tasks have released PPO or GRPO configurations; compare sampler, reward/done/mask, group construction, optimization epochs/LR, global batch, horizons and evaluation cadence against our actual resolved configurations. Then qualify/evaluate the saved learned endpoints under a matched inference protocol if the initial-policy findings support doing so and the cumulative budget allows it. Do not label the shared PPO-derived Sidney configuration as an exact released pi0.5 GRPO recipe or a VLRLA numeric reproduction.

Model computation runs only inside the allocation. Deploy the reviewed latest main commit through Git into a fresh detached source tree; retain original runtime/cache identities. Monitor advancing episodes and every20minutes through terminal status, then close actual scheduler and conservative charges.
