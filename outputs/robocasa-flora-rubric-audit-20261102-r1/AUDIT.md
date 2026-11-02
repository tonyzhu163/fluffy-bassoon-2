# Native completion criteria and a candidate visual rubric

The candidate uses generalized, observable language. Exact simulator distances,
angles and percentages below are human reference material. They are not text
for the video model to measure. The source is the pinned RoboCasa checkout at
3443cdf3dd20f9ae8ed41adba442059b506891d0, including inherited fixture and utility
methods. [Source functions and hashes](NATIVE-CRITERIA-EVIDENCE.json).

| Task | Actual native checks | Consequence for the visual rubric |
| --- | --- | --- |
| OpenDrawer | Target drawer's normalized opening≥0.95 at the final step. | Correct drawer, fully open near its travel limit at the end; partial pull is incomplete. |
| CounterToCabinet | All object bounding-box corners inside the cabinet interior under the helper's geometric tolerance; acting gripper distance from object>0.25m. | Object fully inside and hand clearly withdrawn. Do not require cabinet-door closure. |
| DrawerToCounter | Object contact with any counter fixture; acting gripper distance>0.25m. | Object on a counter and hand clearly withdrawn. Do not require drawer closure or an exact counter patch. |
| TurnOffStove | The named burner's knob is outside the native on interval,0.35≤abs(angle)≤2π−0.35. Cookware can be on another burner. | Judge the named burner, not whichever burner has the pot. Do not require all burners off or hand withdrawal. |
| CoffeeSetupMug | Object-to-dispenser placement-site horizontal distance<0.04m and vertical difference<0.10m; gripper distance>0.25m. | Mug directly under the dispenser and hand clearly withdrawn. Brewing, power-on, uprightness and tray contact are not separate checks in this predicate. |
| CloseFridge | All relevant refrigerator-compartment door joints≤0.005 of normalized travel by default; freezer compartment is separate. | Doors flush closed; visibly ajar is incomplete. This is not the drawer-close0.05 threshold. |
| OpenCabinet | All requested cabinet door joints≥0.90 by default. | All required leaves open, not merely one of a two-leaf target. |
| TurnOnMicrowave | On state plus gripper-to-start-button distance>0.15m. | Appliance on and hand withdrawn from the button. A separate fully-closed-door predicate must not be invented; native power-state logic handles the door. |
| TurnOnSinkFaucet | Water-on state, from the native handle interval0.40<angle<π. | Water flowing; spout direction, pressure and temperature are not additional requirements. |
| OpenStandMixerHead | Normalized head state>0.99. | Head fully raised; release-button contact or a partial lift is incomplete. |

These conditions are task-specific. A global “release and retreat” rule would
incorrectly penalize valid OpenDrawer or TurnOffStove outcomes. Likewise, a
global “close everything afterward” rule would change the pick-and-place tasks.
The mug predicate does not independently test uprightness, stability duration,
tray contact or brewing, so those must not silently become new hard conditions.

Final-state and target grounding should be shared across tasks: maintain the
target from the native instruction and initial scene; do not reinterpret left
and right independently in each camera; distinguish an end state from transient
progress. Use all views, and avoid treating occlusion as automatic failure or
assuming an unseen finish from arm motion alone.

The official scorer already asks for terminal success, so this is a clarification
of endpoint and task semantics, not discovery of a wholly absent terminal
instruction. Its uniform64-frame sampler already includes the first and final
frames. No camera or sampling change is proposed.

The model explicitly answered `Score: 5` on58/183 failed F-training episodes and
196/201 successful episodes. Separately,23 failures received normalized expected
reward≥0.75. `(E[grade]-1)/4` is not `P(full success)`: improving rubric wording
and changing reward aggregation are distinct hypotheses. The candidate preserves
the current score-to-reward mapping for a clean prompt comparison; an alternative
mapping would need its own declared ablation. [Grade evidence](../robocasa-ppo-reward-gap-investigation-20261102-r1/GRADE-CONFUSION.json).

The [candidate prompt](CANDIDATE-RUBRIC.md) is unqualified. Compare it with the
official prompt on exactly the same complete clips, retaining all failures and
ambiguous views. Hold model, video bytes, cameras, frame indices, horizon and
score parsing fixed. Report AUROC plus complete-success false positives,
precision/recall, and low-scored true successes; AUROC alone can hide this issue.
Prompt development cases remain development evidence. Freeze the prompt before
fresh validation or a new PPO arm, and keep any horizon change as a separately
identified factor.

Version the rendered prompt and criterion text in the job/protocol digest and
ready-label receipts. Old prompt labels and cached streaming grades cannot
qualify the new rubric. Static task definitions are allowed; per-episode simulator
truth, joint positions and hidden object poses must stay outside the prompt.
No active run, prompt, cached label, model weight or task-admission decision was
changed by this audit. No scoring inference or new simulator episode was run.
