# Scale references that affect the next steering experiment

Metadata summary,2026-10-30; no new experiment, allocation or corpus audit.
Numbers describe the cited paper/code or existing saved run records. A
steering decision, episode, trial, optimizer step and simulator action are
different units. Realized exposure is distinguished from nominal capacity.

| Process | Tasks / actual or reported source size | Optimization | Motor / steering interface |
| --- | --- | --- | --- |
|VLRLA simulation teacher |Six source tasks; example configuration has204,800 nominal chunk slots/run. Actual valid slots, episodes and per-task manifest counts are unreported. Derived capacity10.24M primitive slots, not executed actions. |25 PPO iterations;8,192 slots/iteration, five epochs, global minibatch2,048. |Action expert updated, VLM frozen; H50/action14; Flow-SDE noise0.3. Not frozen-motor latent-noise RL. |
|VLRLA bridge student |Same simulation histories; exact simulation fitting duration unknown. Separate real-fruit example391episodes/4,692decisions. |Global batch64; real-fruit1,680updates; simulation update count unknown. |Frozen expert with zero-initialized KV residual; six layers/width512. |
|Original DSRL LIBERO |One target goal, three seeds. Motor SFT:40tasks/1,693demos. Paper≈10k online samples; distinct scenes/episodes unknown. |Replay>500 chunks; batch256;UTD20. Released launcher ceiling500k gradient updates, not500k actions. |Frozen pi0;32D repeated across H50, execute20; bound1; authors' pixel SAC. |
|Our LIBERO source SAC |All40train:527K4cases/2,108episodes/326,277actions. Independent8goal validation:584episodes/85,301actions. |233,580 aggregate train SGD, mean5,840/task; validation85,720. Batch256/UTD20/>500 warmup. |Frozen OpenPI pi0.5 H10/execute10;32D tied noise. Pixel actor;4104D prefix separately recorded. |
|Our LIBERO AD/DPT |Same all40 corpus reused by four full/static models; source cost counted once. |500updates/model, batch8, random windows≤128,128×2layers, LR3e-4. |4104D prefix→32D noise. Source25/100/500 exports available. |
|Our LIBERO recurrent PPO |Full/reset:171,166/171,707actions;238/237completeK4trials, respectively. |238/237updates;476/474optimizer steps. OneK4trial/update, two epochs,LR1e-4. |Same prefix/H10/tied32D interface; old source horizons400. |
|Our RoboTwin production recovery snapshot |Six source tasks; full/SAC/reset229/236/236completeK4trials,366,408/394,231/376,136committed source actions; prefixes separately retained. |PPO38/39updates and190/195optimizer steps: sixK4trials/update, five epochs,LR1e-6. SAC244,820SGD, batch256/UTD20. |Frozen Motus H32; full1024D noise, bound1.5,2062public features. Native source cutoffs600/500/400 by task. |

VLRLA facts: [Appendix A.1/A.2](https://arxiv.org/html/2610.19517v1#A1.SS2.SSS1).
The10.24M figure is25×4×256×8×50 nominal slots for the documented source
configuration, not a measured six-task dataset total. Simulation student
duration cannot be filled from the real-robot setting.

DSRL facts: [paper §5.5/C.5/Table11](https://arxiv.org/html/2506.15799v2#A3.SS5),
[released LIBERO launcher](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_libero.sh),
[collection/update units](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L97-L190).
The≈10k paper axis stays in the authors' online-sample unit; it is not converted
to10k primitive actions. Fresh target DSRL is distinct from source meta-training.

Our LIBERO snapshot: [verified evaluation](../../docs/frozen-vla-libero/EVALUATION.md),
[completed source receipts](../libero40-bulk-approved-20261030-r1/terminal-source-receipts.json),
[all40 corpus/training](../libero40-bulk-approved-20261030-r1/README.md),
[coordination at21:35](../coordination.md). Aggregate valid chunk count is not
supplied by that compact terminal receipt; do not infer it by actions/10.
RoboTwin snapshot: existing owner protocol and22:05 recovery audit under
`outputs/robotwin-wholetrial-20261030-r1/PROTOCOL.md` and
`outputs/robotwin-native-failure-20261030-r1/RECOVERY-COST-AUDIT.md` in the
RoboTwin owner's checkout. Production was stopped for simulator reset/GIL
repair, with trained states preserved; the table is not a current liveness claim.

The later human allowance of roughly1M source actions **per learner** is a
future collection design, not an observed result or a rewrite of these older
combined-budget runs. More actions are not assumed to fix a negative result.

## Targeted contrasts before a large expansion

1. Finish the canonical target-DSRL onset curve. With H10 execution and400-action
   episodes,>500 chunks needs at least13maximal episodes; original execute20
   needs26. Fifteen episodes/2,400SGD is onset, not convergence. Keep the pixel
   versus prefix representation difference explicit.
2. Separate fitting scale from data amount: same all40 corpus/model, fixed
   500/1,680/5,000-update endpoints, matched full/static controls and actual
   valid-token exposure. The1,680 endpoint is only a real-VLRLA reference;
   its simulation student schedule is unknown.
3. For RoboTwin, compare tied32D and full1024D steering before10M actions.
   Keep motor/features/tasks/trials/optimizer fixed and declare the new noise
   map. Current full1024D learning is not the published DSRL action space.
4. Compare native Gaussian plus a zero-initialized learned residual against
   direct noise prediction. Constant zero noise is not native behavior. This
   targets current K0 degradation without presuming extra data fixes it.
