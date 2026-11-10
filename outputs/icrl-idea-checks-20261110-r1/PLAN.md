# Three checks of the constructed-history idea, November 10

Development checks for E1 in [fable.md](../../fable.md), on frozen public Human300 π0.5 with the native closed-loop protocol (predict 50, execute 5, ten denoising steps, 600 controls). The scripts were written on macOS and have only been compiled; the first Baige run of each is also its software qualification. No confirmation seeds are used. Nothing here is launched.

## What the existing data already says

The audited Stage 1 AD readout ran six attempts on each of 50 scenes per task. With history cleared, success is barely a property of the scene: the intraclass correlation is 0.05–0.16 across five tasks and both interfaces ([script](stage1_repeatability.py)). Most scenes succeed on some attempts and fail on others. A minority fail every time: 22–23 of 50 on TurnOnMicrowave, 4–7 on CloseToasterOvenDoor and CoffeeSetupMug, 2–3 on OpenDrawer.

This limits what cross-episode history can add on native tasks, and it locates the opportunity in scenes that fail repeatedly. Two caveats: the policy was the AD bridge, not native SFT, and each attempt restarted the same seeded scene.

## Check 1: is failure a property of the scene? (`persistence`)

Four tasks × eight scenes × four motor-noise seeds from the same seeded start, native SFT, no intervention. 128 episodes, about two GPU-hours.

- **Read:** per task, the success rate, the count of scenes with 0/4 and 4/4, and the intraclass correlation.
- **Use:** tasks with many always-failing scenes are candidates for history to matter; tasks near binomial are not. Pick the next task from this table.
- Check the task identifiers against the registered RoboCasa names before launch; `CloseDrawer` is unverified.

## Check 2: can a known correction rescue a natural failure? (`rescue`)

On Nov 10 the policy failed six of eight OpenDrawer scenes. Five never grasped the handle and held a steady lateral miss of 1–4 cm at the approach. For each failing scene, rerun with a lateral goal offset of −4, −2, +2 or +4 cm held from the approach until the first grasp, at most 100 controls. Up to 32 episodes, about 30 minutes.

- **Read:** per scene, whether any offset gives native success, and whether the rescuing sign opposes the nominal miss.
- **Go:** an offset rescues at least three of the six, with a consistent sign rule. That gives a correction worth inferring from a failed attempt.
- **No-go:** no rescue, or rescue with no relation to the miss. The frozen policy then fails for a reason a lateral correction cannot fix; inspect the videos before trying another actuator.
- The nominal rerun must reproduce the Nov 10 outcome on each scene. If it does not, closed-loop reruns are not deterministic and the comparison is invalid.

## Check 3: a hidden property revealed only by contact (`stiff`)

On the two scenes that open (982101, 982103), scale the drawer slide joint's damping by 8 and 64. Run two noise seeds each, with and without doubling the commanded translation after the first grasp. 20 episodes, about 15 minutes.

- **Read:** drawer travel and success per damping level; agreement between the two noise seeds; whether doubled effort recovers success.
- **Go:** some damping level fails on both seeds without extra effort and succeeds with it. That is a persistent, outcome-revealed property with a working correction.
- **No-go:** the policy opens the drawer at every level, or nothing recovers it. Change the property (friction, required travel, payload) before building any corpus.
- Record the reported joint damping in the result; the script sets it to at least 1 before scaling.

## Launch

On an allocated Baige GPU container, from a clean checkout of the commit that contains this folder:

```sh
bash outputs/icrl-idea-checks-20261110-r1/run.sh <commit> <fresh-id> rescue
bash outputs/icrl-idea-checks-20261110-r1/run.sh <commit> <fresh-id> stiff
bash outputs/icrl-idea-checks-20261110-r1/run.sh <commit> <fresh-id> persistence OpenDrawer,CloseDrawer,CloseFridge,TurnOnMicrowave 8 4
```

Run `rescue` and `stiff` first; they are short and decide the actuator. The launcher is the Nov 10 drawer launcher with the output folder and entry point changed. Total is under three GPU-hours on one Pro6K.
