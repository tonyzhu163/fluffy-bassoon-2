# Candidate visual completion rubric

Draft for validation. Only the shared text and the current task's completion
text are intended for the scorer. Physical thresholds and simulator outcomes
belong in the separate human audit, not in the rendered prompt.

Use the native task instruction to identify the requested object and target.
Follow the same target throughout the video, even when camera viewpoints change.
Use the synchronized views together. Judge whether the requested outcome is
achieved at the end of the trajectory. Earlier progress that is later undone
does not establish terminal success.

Reserve complete success for an outcome satisfying all of the task's completion
conditions. Reaching, touching, grasping, moving toward the goal or nearly
finishing are partial progress. Use visible evidence; occlusion alone is neither
proof of failure nor proof of success. Do not add requirements absent from the
task's completion conditions. Keep the official one-to-five output format.

| Current reward task | Observable completion text |
| --- | --- |
| OpenDrawer | The requested drawer is fully open at the end, near the end of its travel. A partly pulled drawer is not complete success. Judge the drawer identified by the instruction and initial scene, rather than switching targets as the camera moves. |
| PickPlaceCounterToCabinet | The requested object has been placed fully inside the intended cabinet, and the acting hand has released it and withdrawn clearly away. Holding it in the doorway or leaving it protruding is incomplete. Closing the cabinet door afterward is not required. |
| PickPlaceDrawerToCounter | The requested object has been taken out of the drawer and placed in contact with a counter surface. The acting hand has released it and withdrawn clearly away. Keeping it held above the counter is incomplete. Closing the drawer afterward is not required. |
| TurnOffStove | The burner named in the instruction is off at the end. Follow the named burner and its control; cookware may be on a different burner. Other burners do not have to be turned off. Moving a knob without reaching the requested off state is incomplete. |
| CoffeeSetupMug | The requested mug is positioned directly under the coffee dispenser at the receiving position. The acting hand has released it and withdrawn clearly away. Leaving it beside the machine or holding it under the dispenser is incomplete. Brewing coffee or turning the machine on is not required. |

The following tasks are evaluation-only references, not additions to PPO's
reward-training task set:

| Evaluation task | Observable completion text |
| --- | --- |
| CloseFridge | All doors of the refrigerator compartment being closed are flush closed at the end. A door left visibly ajar is incomplete. Do not add a separate freezer-door requirement when the task concerns the refrigerator compartment. |
| OpenCabinet | The requested cabinet is fully open at the end, with all of its required door leaves open. Opening only one leaf of a two-door target is incomplete. |
| TurnOnMicrowave | The microwave has entered its on/running state after the start action, and the acting hand has moved clearly away from the start button. Touching or continuing to hold the button does not by itself establish completion. |
| TurnOnSinkFaucet | Water is flowing from the requested faucet at the end. Merely moving the spout or touching the handle is incomplete. No additional temperature or pressure setting is required. |
| OpenStandMixerHead | The mixer head is fully raised at the end. Touching the release or lifting the head only partway is incomplete. |

Do not apply the hand-withdrawal condition universally: drawer opening and stove
turn-off do not require the hand to retreat under their native success checks.
