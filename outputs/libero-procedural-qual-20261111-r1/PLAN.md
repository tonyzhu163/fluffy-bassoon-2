# Development qualification before motor SFT

User approved this next step on November 11, 2026: qualify generated development
scenes, measure released π0.5 with correct goals supplied, and use the result to
decide whether motor SFT is needed. This is separate from the ongoing BPP prompt
retargeting study. No hidden-rule learner or SFT job is launched by this plan.

The development subset is fixed before simulation: the lowest eight indices in
the case bank's training layout split, `[0, 1, 2, 5, 6, 10, 11, 13]`. Validation
and test layouts remain unused. [SCENES.json](SCENES.json) contains the exact
generated descriptions and task-file hashes, with no selection by policy result.

First run the unchanged qualified BPP simulator runtime in a fresh allocation,
using 256-pixel observations to support the later π0.5 check. Record construction,
initial contacts, settling displacement, visibility, initial goal predicates and
the native results of dropped-bowl predicate controls. Those controls inspect
goal semantics, not robot reachability. Preserve every rejection and error.
Record numerical geometry diagnostics without unsupported drift/spacing cutoffs.
Recompute all four-axis rule labels from actual settled object/fixture positions;
nominal layout labels must not be mistaken for realized reset labels.

Use the released π0.5 Spatial Combination checkpoint pinned in [SOURCE.json](SOURCE.json),
with its author's JAX configuration, image transforms, checkpoint normalization
and five-control replanning interval. Establish native-scene execution first,
then inspect a small generated-scene panel with unambiguous concrete goal prompts.
For duplicate bowls, the prompt must resolve the intended bowl in the actual view;
the prior release's neighbour phrases cannot simply be reused after moving objects.
Keep instruction-grounding errors distinguishable from failed grasps/placements.

The first policy panel is four native episodes (the two prior native-control
tasks, their own released initial states 0 and 1), then both bowls to the plate
in each usable generated layout. Generated prompts say left/right in the actual
mirrored policy image; this is an instruction-grounding diagnostic and does not
by itself prove or disprove motor competence. The plate panel does not establish
coverage of every destination in the rule family. Preserve skipped geometry
cases and all policy failures. Review traces before expanding destinations or
attributing failures to missing motor skills.

`prepare_pi05.sh` installs the author's frozen CUDA 12 lockfile into a separate
Python 3.11 environment and verifies the pinned checkpoint file sizes/hashes.
The simulator remains in its existing environment; a separate client dependency
directory supplies the official websocket client. `launch_pi05.sh` probes the
allocated GPU, runs the JAX server on loopback, records memory, and caps each
policy allocation's processes to about one hour. Use compatible hardware for
that CUDA 12/JAX release; Blackwell requires a separately qualified runtime.
These new policy scripts have only static checks as of preparation.

Report native outcome for every episode, intended/actually moved bowl, destination,
controls, latency and GPU memory. Use these descriptive results to decide whether
the next intervention is interface repair, better demonstrations, or motor SFT.
No blanket percentage threshold and no held-out filtering by success.

Use one allocated GPU at a time, with the actual job-step probe, immutable Git
deployment and fresh output identity for each attempt. Initial qualification
budget: at most 4 GPU-hours including failed jobs. CPU preparation runs on the development endpoint when available; any
time spent preparing inside a GPU allocation counts toward this cap. Any model inference requires the actual accelerator allocation; a CPU import
is not accelerator qualification. Save clocks and cumulative consumption here.
