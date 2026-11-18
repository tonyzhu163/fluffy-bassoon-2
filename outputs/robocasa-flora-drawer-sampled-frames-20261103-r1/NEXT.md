# Next small rubric screen: exact boundary and two separate target hints

The sampled-frame audit supplies opening evidence for the missed128 success
and near-closed upper-front evidence for wrong-drawer112/376. This justifies
testing initial target identity and drawer type, without changing cameras or
adding per-case answers. Precise extent remains a separate difficulty.

Freeze the3 unqualified texts in NEXT-RUBRICS.json before inference:

- native_boundary changes only1.6's 'roughly half its front-to-back depth'
  to 'at least52.25% of its front-to-back depth, just over half'.
- initial_target_lock adds only: 'Identify this drawer front in the initial
  frames and keep tracking that same front throughout the video.'
- shallow_target_type adds only: 'The target can be a shallow drawer directly
  above cabinet doors.'

The hints are separate comparisons to the same exact-boundary base. They are
general task descriptions, not episode IDs, colors, object names or simulator
answers. Do not combine both hints or retune score cutoffs after seeing grades.
The source-correct base matters: a hint added to approximate1.6 alone cannot
establish fidelity to the near-boundary simulator criterion.

Use26 retained causal prefixes from the existing10-video panel for an early
screen:112@300/350, and256/336/376@700/750; for each successful64/72/128/144/
216/264, use the50-control grid point immediately before native first success,
the next point, and the following point. Thus14 negative and12 post-success
endpoints,4 never-success and6 ever-success cases. Score3×26 candidates plus
the same4 exact1.6 repeats; recover26 reference labels:82 new outputs,104 paired
labels plus4 repeats,13 batches of at most8. Source clips and labels already
exist; no new trajectory or motor call is required.

Keep the existing primary two-Score5-probes/50-control-spacing rule, accepting
the second and latching. Require0 false Score5 endpoints or false accepted
episodes,6/6 native successes accepted after their crossing and within100
controls, and4/4 repeated references. Report >=4 only as a diagnostic. A pass
on this early screen must then face the complete45-prefix small panel and a
new small retained-case validation set. It is not production qualification.

Before launch, create a fresh output identity, review latest main, deploy a
clean immutable Git pin, qualify actual Linux prompt/video reads and rendered
messages, and check live Baige capacity/hardware. The intended one-GPU scope
remains32CPU/128GiB RAM/16GiB SHM,15minutes, auto retry off and at most900GPU-s,
with in-step GPU probing and finite deadlines. The preceding94-output/13-batch
run used643 workerGPU-s; prior6676 workerGPU-s remain separately charged.
This file is an unlaunched design, not a launch receipt or a qualified result.
