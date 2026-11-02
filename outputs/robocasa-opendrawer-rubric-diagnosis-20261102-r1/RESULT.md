# OpenDrawer failure diagnosis from retained actions

Three CPU action replays reproduce all15 saved normalized public16D robot
states exactly (max absolute difference0) and match every retained native scene
metadata field. Their terminal simulator outcomes also reproduce. No full
MuJoCo initial state was saved; these are strong identity checks, not a saved
full-state restoration.2250 retained controls were physically replayed on
allocated Baige dev CPU, no motor calls, newly sampled actions, GPU inference
or learner updates.

| Retained case | Target behavior | First native success | Terminal |
| --- | --- | --- | --- |
|112|Target uppermost stack_1_main_group_4 stays shut (normalized~0.00001); drawer immediately below, stack_1_main_group_3, opens to1.817|Never|Failure|
|64|Correct target opens to maximum1.281, then is pushed back to0.8205|Control542 /27.1s|Failure|
|144|Correct target finishes at1.812|Control222 /11.1s|Success|

The frozen native criterion is>=0.95 normalized, where normalization divides
outward distance by0.55×fixture depth. The actual threshold is0.5225×depth.
“95% of mechanical travel” was inaccurate. Details and exact source exports
are in NATIVE-CRITERION.md, SOURCE-SURVEY.json and REPLAY-SOURCE.json.

The fourth planned example, case128, was stopped before policy controls because
object_cfgs differed. Inspection locates the mismatch in the order of the
semantic tags `tool` and `receptacle`, but the frozen strict qualification is
not silently relaxed. All other metadata and the initial robot state match.
No drawer-state or timing conclusion is drawn for that example.

Case112 supports an explicit uppermost-drawer identity intervention. Case64
supports separating terminal endpoint grading from first-success grading:
it is a terminal failure but would succeed under stop-at-first-success.
This tiny diagnosis does not estimate how common either mode is in the183
failures, and does not establish a first-success precision or timing result
for Flora. New rubric candidates are tested only on the existing14-video
failure-focused panel in ../robocasa-flora-rubric-failure-sample-20261102-r3/.

The initial launch used the LIBERO-only environment and failed on import,
before simulator construction or output creation. The corrected launch used
the qualified RoboCasa environment and completed the finite diagnosis.
