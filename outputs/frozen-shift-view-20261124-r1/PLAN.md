# Body view: does picturing the source arm for the whole episode rescue a Kinova3 failure?

Lane of the [proposal-shift program](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). Written before any result of this lane. Policy weights and motor parameters never change.

## Question

Frozen π₀.₅ proposals move when a Panda arm is drawn at the Kinova3 grip site. Does drawing it at every replan of an episode turn a Kinova3 timeout on layouts 14, 15, 17, 36, 37, 38 into native reward, without losing 16, 31, 34, 35, 39?

## Prediction

The Kinova re-render reproduces the untreated branch command for command. Panda in the external camera changes the external image digest at every replan and nothing else in the policy input. If arm appearance is what holds the policy on the failing behaviour, it rescues failures beyond the untreated repeat's discordance.

## What counts against it

- No rescue on the six failures, or losses on the five successes match rescues.
- Kinova re-render differs from raw in any command: the render path is not a clean control and the Panda branches are uninterpretable.
- CPU, before native: the Panda composite hides task objects that the Kinova frame showed at a scale comparable with the arm edit itself. Then a native effect cannot be attributed to arm appearance alone.

## Prior work checked

- [Source view](../frozen-vla-source-view-20261123-r1/NATIVE.md): Panda render changed 10/10 proposals on archived frames; native on layouts 16 and 17 only, 0 rescues, 0 losses, +78 controls on 16; zero wrist pixels changed on 219 boundaries. Qualified renderer `source_view.py` (`Sibling`, `compose`, `Exposure`) is reused unchanged.
- [Rendered appearance](../frozen-vla-factory-20261123-r1/rendered-appearance/RESULT.md): on SmolVLA and UR5e, a Panda sprite from public IK halved translation-proposal drift offline; a displaced sprite made it worse. Offline forward pass only.
- [Link painting](../frozen-vla-visual-adapter-20261123-r1/RESULT.md): colour alone did not change success.
- Mirage (cross-painting at test time) is the method this follows; it also uses a privileged renderer.

## Protocol

Adapters, all `harness.Adapter` subclasses in `adapter.py`:

| Name | Policy input change |
| --- | --- |
| `view_kinova` | Kinova arm pixels replaced by an independent Kinova render. Control for the render path |
| `view_panda_external` | Panda at the grip site in `agentview`. Wrist image untouched |
| `view_panda_both` | Same, plus the same z-tested substitution in `robot0_eye_in_hand` |

Wrist treatment. To be settled from archived evidence in step 1: if the Panda wrist camera at the same grip site has the same pose and intrinsics as the Kinova one and neither arm is visible in it, the faithful source wrist view equals the target wrist view and no other wrist edit is justified. `view_panda_both` then only differs from `view_panda_external` at replans where an arm link enters the wrist frustum; the per-replan `inputs_changed` record says when.

Renderer. The harness gives hooks public observations only, and the shared panel owns the simulator. The adapter therefore owns a render twin (`twin.py`, simulator runtime): a second Kinova3 environment identified by byte-equal public reset observation among the registered starts, advanced with the issued commands from `observe`, plus the Kinova and Panda render siblings. Before every render the twin's own public observation must equal the one the harness passed, byte for byte, or the hook raises and the panel drops that start for that adapter. The twin's scene, depth and object state are privileged simulator knowledge: this is a diagnostic adapter, not a deployable one. Nothing from the twin other than the two images reaches the policy. On a Panda start the adapters are the identity.

Steps:

1. CPU, archived frames (`analyze.py`, ten Kinova3 exposures of layouts 14 and 15): per camera, pixels changed at 256 and after a 224 resize proxy; split of the changed pixels into Kinova arm removed, Panda arm added over scene, and scene revealed; Panda pixels that a depth-free paste would draw over nearer scene geometry; newly hidden scene pixels that are not table or wall by colour, as the task-object overdraw count. No segmentation is archived, so the colour split is a proxy and is labelled as one.
2. CPU fixture (`test_view.py`): with a fake renderer, each adapter changes only the image digests it owns, the twin is advanced with exactly the consumed commands, a twin mismatch is a recorded hook fault, Panda starts are untreated.
3. Native, not run here: `run.sh <pin> <attempt>` runs the shared panel with the three adapters on the development starts. Report successes, rescues, losses beyond repeat, controls, wrist-changed replans and twin faults. A rescuing adapter goes once to layouts 40–47 and Panda 0–3.

Evidence classes: step 1 is archived replay frames, step 2 is CPU fixture, step 3 is native reward and is the only one that can show task success. `twin.py` needs MuJoCo and is not executable on this VM.
