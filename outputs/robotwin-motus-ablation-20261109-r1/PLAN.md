# Motus Released π0.5 2x2 SFT Baseline Ablation (`place_phone_stand`)

User instruction (2026-11-09): on a separate git worktree, test both SFT baseline candidates for `motus-robotics/pi0.5_robotwin2` (`effc6e1a99bd4d0edf9447e8ba313def67649cfe`) on `place_phone_stand` (`demo_clean`, native cuRobo `expert_check` starting at seed `100000`) to establish the SFT baseline level before running PPO and GRPO:

1. **Prompt format (`raw` vs `default_prompt`):**
   - `raw`: unseen task instruction string from `generate_episode_descriptions` (e.g. `"Set the rectangular phone on the silver arm phonestand."`).
   - `default_prompt`: prepends the exact 37-word `default_prompt` saved in `motus-robotics/pi0.5_robotwin2/metadata.pt` (`LeRobotAlohaDataConfig.default_prompt` from `thu-ml/Motus/data/robotwin2/robotwin_data_convert/config.yml`):
     `"The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: " + instruction`

2. **Trajectory execution (`take_action` vs `continuous_chunk` `TOPP`):**
   - `take_action`: native RoboTwin `Base_Task.take_action` (step-by-step 2-point `TOPP` on each of the 32 waypoints, stopping at every waypoint).
   - `continuous_chunk`: continuous 32-waypoint chunk `TOPP` (`compress_path(atol=1e-3)` + `mplib.Planner.TOPP(path, 1/250)` without `downsample_trajectory` velocity distortion, synchronous stationary-arm hold so a stationary arm never inflates step count, and 15-physics-step/waypoint linear fallback matching RoboTwin's `save_freq = 15` demonstration collection rate if spline parameterization fails, checking `check_success()` at every physics step).

## 2x2 Paired Design

Every candidate seed starting at `100000` is screened once with native `setup_demo` + `play_once()` (`cuRobo` expert planner). For each expert-feasible seed, a single unseen instruction is drawn and all four arms are evaluated on the exact same `(seed, instruction)` pair with matched per-decision initial flow-matching noise (`torch.manual_seed(seed * 1000 + decision_idx)`):

1. `raw_take_action` (Control)
2. `default_prompt_take_action` (`DEFAULT_PROMPT` + `take_action`)
3. `raw_continuous_chunk` (`raw` + continuous-chunk `TOPP`)
4. `default_prompt_continuous_chunk` (`DEFAULT_PROMPT` + continuous-chunk `TOPP`)

Decision-0 predicted action chunks are asserted bit-identical between `(raw_take_action, raw_continuous_chunk)` and between `(default_prompt_take_action, default_prompt_continuous_chunk)` so controller comparisons at decision 0 have zero policy-sampling variance.

## Qualification and execution

Before launch, rebased onto current `main` (`329fcded`) and adopted the isolated
`motus-reference-openpi-20261109-r2` runtime. The earlier r1 runtime produced
OIDN errors on Blackwell; r2 passed native clean/randomized expert rendering
with OIDN 2.3.3 ([renderer evidence](../robotwin-motus-release-baseline-20261109-r1/RENDER01-READBACK.json)).
The ablation pins the OIDN loader and libraries in its CPU manifest and rejects
denoiser errors. Native simulator and policy settings remain unchanged.

Run a two-pair GPU qualification, then ten paired seeds (40 policy episodes)
after the qualification passes. The qualification is operational evidence;
the ten-pair evaluation is the diagnostic comparison. Each allocation is one
GPU for at most 60 minutes, automatic retry disabled; reserve at most two
GPU-hours including startup. Record exact source, scheduler identities,
completed pairs and actual usage in this run directory and the own coordination
row. Larger sample sizes require a separately recorded decision; these small
samples establish feasibility and diagnostic baseline levels only.
