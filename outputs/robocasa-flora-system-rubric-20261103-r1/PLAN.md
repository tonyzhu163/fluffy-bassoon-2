# Put the rubric in grading instructions and align first-success messages

The preceding two task-text rewrites fail on the fixed six-video panel.
All tested task-text variants retain the generic system instruction calling
the model a terminal-success rewarder and the final user request to evaluate
terminal success. That observable mismatch motivates this experiment; it does
not establish the model's internal cause.

Keep the same six retained videos,35 prefixes, three-view collage, duration-aware
official sampler, model/processor/tokenizer, seed,1–5 output grammar and parser.
No new cases, robot trajectories, simulator episodes, motor calls or PPO.

1.11 preserves the native short task in the user message and puts the verbatim
1.6 side-specific criterion in the system grading message. This changes rubric
placement and restores the native task as context; do not claim an isolated
role-only effect. Its grading instruction remains terminal.

1.12 preserves the native short task and puts the verbatim1.10 criterion in
the system message. Its generic system description and final user question
also explicitly ask whether success occurred anywhere in the observed prefix.
This is a composite alignment change, not a single-clause attribution.

These are custom grading messages using the same official terminal head,
output grammar and numerical parsing, not the exact official prompt. The
production worker source stays byte-identical. The run-local adapter selects
messages in its own process and restores the original functions after use.
Store prepared and actual message objects and compare them by hash. Instrument
the actual renderer call to require each scoring request to receive the
prepared rubric, and record the rendered-prompt hashes. No simulator outcome,
joint state, first-success time, seed or case ID enters the model prompt.
Only the task's static completion rule and native left/right instruction do.

Recover140 verified reference grades under1.1/1.6/1.9/1.10. Generate35 per new
criterion plus four verbatim1.6 anchors:74 new outputs,210 candidate/reference
grades and4 separate repeats. Freeze the same development screen:4 anchor
grades reproduce;0 Score5s on20 sampled pre-success/wrong-target prefixes;
all5 successful episodes first receive a sampled5 after native first success
and within100 controls. A pass still requires a denser clock check and new
small validation panel before any production qualification.

Use reviewed latest main and a new immutable Git pin. On Linux CPU, verify
frozen input/video hashes, source identity, public-job schema, native task
preservation, all74 prepared messages, actual50/100/750 prefix decoding and
official24/50/64 sample counts. Private references stay outside scoring.

One Baige NVIDIA RPBZZZ6 GPU,32 CPU/128GiB RAM/16GiB SHM,15-minute maximum,
auto retry off; actual allocation GPU probe required. TP1/batch8/.78 memory,
temperature0, seed0, finite780/810/860-second internal deadlines, cap900 GPU-s.
Prior1876 measured prefix-attempt GPU-s remain separate. No automatic retry,
full634-video scoring, production rubric/cache change or training launch.
