# Three checks of the constructed-history idea, November 10

Development checks for E1 in [fable.md](../../fable.md), on frozen public Human300 π0.5 with the native closed-loop protocol (predict 50, execute 5, ten denoising steps, 600 controls). Use the first allocated Baige episodes to qualify each intervention. All seeds are development seeds.

## What the existing data already says

The audited Stage 1 AD readout ran six attempts on each of 50 scenes per task. With history cleared, success is barely a property of the scene: the intraclass correlation is 0.05–0.16 across five tasks and both interfaces ([script](stage1_repeatability.py)). Most scenes succeed on some attempts and fail on others. A minority fail every time: 22–23 of 50 on TurnOnMicrowave, 4–7 on CloseToasterOvenDoor and CoffeeSetupMug, 2–3 on OpenDrawer.

This limits what cross-episode history can add on native tasks, and it locates the opportunity in scenes that fail repeatedly. Two caveats: the policy was the AD bridge, not native SFT, and each attempt restarted the same seeded scene.

## Check 1: is failure a property of the scene? (`persistence`)

Four tasks × eight scenes × four motor-noise seeds from the same seeded start, native SFT, no intervention. 128 episodes, about two GPU-hours.

- **Read:** per task, the success rate, the count of scenes with 0/4 and 4/4, and the intraclass correlation.
- **Use:** tasks with many always-failing scenes are candidates for history to matter; tasks near binomial are not. Pick the next task from this table.
- All four Gym identifiers resolve in the pinned Baige RoboCasa environment. A seeded fridge reset differed on its second repeat. Before any actions or rewards on a scene, select the first native start signature observed twice under the registered seed. Match each subsequent repeat to that signature before inference, up to 16 resets per repeat. Record every candidate and its component hashes. The signature covers full flattened simulator state, camera pose, model body and geometry positions, policy state, instruction, and each policy image. Keep all eight prespecified seeds. Report any scene that cannot be matched.

## Check 2: can a known correction rescue a natural failure? (`rescue`)

On Nov 10 the policy failed six of eight OpenDrawer scenes. Five never grasped the handle and held a steady lateral miss of 1–4 cm at the approach. For each failing scene, rerun with a lateral goal offset of −4, −2, +2 or +4 cm held from the approach until the first grasp, at most 100 controls. Up to 32 episodes, about 30 minutes.

- **Read:** per scene, whether any offset gives native success, and whether the rescuing sign opposes the nominal miss.
- **Go:** an offset rescues at least three of the six, with a consistent sign rule. That gives a correction worth inferring from a failed attempt.
- **No-go:** no rescue, or rescue with no relation to the miss. The frozen policy then fails for a reason a lateral correction cannot fix; inspect the videos before trying another actuator.
- The first attempt stopped before any offset branch. On seed 982100 its outcome, duration, first grasp and decision observation reproduced, but terminal physics differed. The repeat below preserves all eight original outcomes as the eligibility gate. Retain approach and wrist videos. Treat one or two rescues as below the threshold for expanding collection.

### Rescue replay amendment after the first baseline

Replay the archived actions for each seed and log the first differing physical state. Independently rerun the native policy and compare its commands and states to the archive. Report exact terminal matches separately for fixed-command and closed-loop replay. Require both runs to reproduce the archived native outcome on all eight seeds before testing offsets. For each of the original six failures, replay the new nominal command prefix for zero and all four offsets. Every physical state through the decision and the public decision observation must match the new nominal. A zero-offset continuation must reproduce its native failure before scoring that scene. This preserves the original six-scene denominator and makes numerical drift visible. The run has 46 episodes including eight archived-tape replays and six additional zero continuations.

## Check 3: a hidden property revealed only by contact (`stiff`)

On the two scenes that open (982101, 982103), scale the drawer slide joint's damping by 8 and 64. Run two noise seeds each, with and without doubling the commanded translation after the first grasp. 24 episodes including doubled effort at ordinary damping.

- **Read:** drawer travel and success per damping level; agreement between the two noise seeds; whether doubled effort recovers success.
- **Go:** some damping level fails on both seeds without extra effort and succeeds with it. For a claim that history is needed, normal effort must also outperform doubled effort on ordinary drawers in a matched comparison. Compare against both fixed effort choices.
- **No-go:** the policy opens the drawer at every level, or nothing recovers it. Change the property (friction, required travel, payload) before building any corpus.
- Record the reported joint damping in the result; the script sets it to at least 1 before scaling.

## Launch

On an allocated Baige GPU container, from a clean checkout of the commit that contains this folder:

```sh
bash outputs/icrl-idea-checks-20261110-r1/run.sh <commit> <fresh-id> rescue
bash outputs/icrl-idea-checks-20261110-r1/run.sh <commit> <fresh-id> stiff
bash outputs/icrl-idea-checks-20261110-r1/run.sh <commit> <fresh-id> persistence OpenDrawer,CloseDrawer,CloseFridge,TurnOnMicrowave 8 4
```

Run `rescue` and `stiff` first; they are short and decide the actuator. The launcher is the Nov 10 drawer launcher with the output folder and entry point changed. The planning estimate is under three GPU-hours. Record measured GPU usage for all checks.
