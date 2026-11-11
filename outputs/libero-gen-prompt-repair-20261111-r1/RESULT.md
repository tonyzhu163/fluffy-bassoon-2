# Successful demonstrations did not repair frozen BPP execution

The paired comparison completed on Baige on 2026-11-11. Original prompts
succeeded in **0/4** policy episodes; repaired prompts also succeeded in
**0/4**. All eight episodes executed the full 562-control horizon and never
reached either bowl-to-plate native predicate. Both repaired demonstrations
themselves reached their intended native goals on layout 0. Changing the
demonstrations alone therefore did not repair this frozen checkpoint on the
two tested development scenes, including the demonstration's own scene.

No weights or history reader were trained. This is a completed negative
prompt-repair result, not a general impossibility claim about prompting,
fine-tuning, other checkpoints, or other layouts.

## Paired policy evidence

The evaluation used commit `ff787a05`, job `job-e16eyoc5er5s`, and one NVIDIA
RPBZZZ6 in the Baige Pro6K queue. It ran from 21:23:44 to 21:28:54 Shanghai;
creation-to-end conservative accounting starts at 21:23:31.

| Scene | Intended choice | Policy seed | Original | Repaired |
| --- | --- | ---: | ---: | ---: |
| Layout 0, demonstration scene | Center bowl | 91001 | 0 | 0 |
| Layout 0, demonstration scene | Bowl near plate | 101001 | 0 | 0 |
| Layout 1, nearby scene with swapped physical IDs | Center bowl | 91002 | 0 | 0 |
| Layout 1, nearby scene with swapped physical IDs | Bowl near plate | 101002 | 0 | 0 |

Scores are the native selected-object goal, not a new numerical admission
threshold. Layout 1 was not used to tune the teacher; it is a small nearby
transfer check, not a broad held-out benchmark. One policy seed per
scene/choice does not estimate a general success rate precisely.

The two conditions had identical initial simulator states, public
observations, selected physical objects, per-episode seeds, and control
horizons. Both processes strictly loaded all **967** checkpoint tensors and
kept the model frozen. The checkpoint SHA-256 is
`74e0f841f3b86f1383adda23dab59fddee8806a4bcb5685ed7f826afb320c097`.

Original prompts were unchanged copies of source demo 0. Loading those
copies through the official dataset path reproduced the historical prompt
tensor hashes exactly (`31f72d80…` and `43196559…`). Repaired prompts had
distinct hashes (`be466a18…` and `04588bf7…`). Both conditions used fresh
one-demo dataset roots without the checkpoint's training-split filter;
released normalization, image preprocessing and delta-action conventions
were retained. Repaired demonstrations were never represented as examples
from the checkpoint's training split.

See [paired outcomes](evidence/paired01/comparison.json), the
[original](evidence/paired01/original/result.json) and
[repaired](evidence/paired01/repaired/result.json) results, and their
[original](evidence/paired01/original/construction-receipt.json) /
[repaired](evidence/paired01/repaired/construction-receipt.json)
construction receipts. Packing receipts record the exact source, output
hashes, action convention, observation alignment and chosen endpoint.

## Behavior changed without improving success

All four matched original/repaired action-array hashes differ. Independent
video and body-trajectory inspection found that the repaired center-bowl
prompt transported the intended center bowl toward the plate in both
scenes, but did not complete placement. The repaired near-plate prompt also
ultimately transported the center bowl, the wrong candidate for that
prompt. Thus the new prompts influenced behavior, but did not reliably
preserve the intended choice or solve placement.

The intended center bowl rose by about 8.6 cm / 5.3 cm with its repaired
prompt on layouts 0 / 1. With the repaired near-plate prompt, the wrong
center bowl rose by about 6.1 cm / 6.3 cm. These are descriptive motion
measurements, not success criteria. Full trajectories include incidental
contacts and movement of the other bowl.

See [motion measurements and action hashes](evidence/paired01/parent-motion-readout.json)
and the per-condition `audit/record-*.json` traces. All eight videos are
retained with artifact hashes. Prompt dependence does not identify the
transformer's internal mechanism or demonstrate reward-history learning.

## Demonstration repair and its limits

The native source demonstrations replayed successfully for both candidates.
Their observation alignment supports post-action records: position RMSE
was 0.406 mm versus 4.981 mm for candidate 0, and 0.0125 mm versus 2.553 mm
for candidate 1, comparing post versus pre. New HDF5 prompts use that
alignment, actual executed OSC delta controls, and the official vertical
image flip / orientation conversion path.

The first retargeting attempt grasped and lifted both intended bowls.
Candidate 0 descended before alignment, pushed the plate, and never reached
its goal. Candidate 1 reached native success at zero-based action 333,
remained successful for several controls after opening the gripper, and
later crossed the native predicate's 3 cm horizontal boundary. Its complete
372-control trajectory, including that later loss, is retained.

For candidate 0, teacher04 replayed the exact successful 133-control grasp
prefix and attempted alignment. It exhausted its 80-control alignment
budget because its controller continually reset a small position error
instead of advancing a persistent target. The wrist drifted downward and
the bowl remained 9.90 cm short horizontally. Teacher05 corrected only that
reference update, retaining the scene, prefix, phase budgets and tolerances.
It entered descent at action index 189 and reached native success at index
261: **262 actual controls**, exact prefix replay, unchanged initial public
observations, and final bowl–plate horizontal distance **5.56 mm**.

The resulting prompts contain candidate 0's teacher05 trajectory and
candidate 1's teacher03 prefix through its first native success (334
controls). Ending at first native success matches policy episode stopping.
The provisional requirement that a longer teacher schedule remain successful
at its final frame was removed because it added a persistence criterion
outside the native benchmark definition. We do not claim stable released
placement beyond the scored endpoint. Full-source final rewards and lengths
remain explicit in the packaging receipts.

See [teacher03 diagnosis](evidence/teacher03/trajectory-diagnosis.json),
[teacher04 recipe](TEACHER04.md), [teacher05 correction](TEACHER05.md), and
[successful center-bowl record](evidence/teacher05/teacher/layout00-0/result.json).

Two earlier allocations failed before simulation: an author-local XML
asset path and an old Git bundle-reader incompatibility during a development
console outage. Their records are preserved as setup failures, not policy
results. No automatic retries or score-percentage gates were used.

## Next decision and retained artifacts

This closes the bounded prompt-only repair. A further motor repair should
use varied successful demonstrations and test motor fine-tuning, or use a
different released motor if its separate development qualification succeeds.
The separately owned π0.5 qualification is relevant to that choice; its
outcomes are not part of this BPP result. More history-reader training would
not by itself resolve the demonstrated execution/choice failures. No larger
panel or new training job was launched automatically.

Repair cost was **649 scheduler / 1,039 conservative GPU-seconds**
(**0.180278 / 0.288611 GPU-hours**), within the initial 4 GPU-hour cap.
Combined prior toy, native motor, custom-layout and repair work totals
approximately **17.3159 scheduler / 18.0359 conservative GPU-hours**.
All repair jobs are finished. Full NPZ/HDF5 artifacts, videos and immutable
checkouts remain under the shared project's run and source directories; compact receipts,
audits, scripts and hashes are committed. Nothing required by another active
job was moved or deleted. [Accounting and job identities](EXECUTION.json).
