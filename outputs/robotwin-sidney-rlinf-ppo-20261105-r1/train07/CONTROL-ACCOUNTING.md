# Source totals are conservative motor-command charges

The train07 source journal safely charges requested native14D motor commands,
including unresolved calls and setup allowances. It does not directly measure
executed motor calls or internal physics integration substeps. For example,
the first immutable rank0 trace has`measured_valid_mask=False` and
`executed_or_conservative=3200` (64environments×50requested commands).
Earlier progress wording calling these totals physical/executed controls is
corrected to requested/conservatively charged controls; numerical ceilings,
optimizer counts and checkpoint/frozen/replay evidence are unchanged.

Pinned RLinf`c605b9d5` takes`venv.step` when skip_intermediate_renders=True,
and creates valid_action_mask only when executed_action_count is provided.
Pinned RoboTwin`dca9ec68` returns from build_result before attaching that count
when collect_obs=False. Its already-successful-task branch can return success
without executing another command. Therefore requested chunk length cannot be
used to infer delivered actions or physical simulation substeps. Native learner
dones/loss masks remain their original values; this audit does not change them.

The bounded engineering and matched baselines claim the declared common native
case/horizon protocol and conservative action ceiling. They do not claim equal
measured executed controls. Source6/student-corpus admission staysFalse until
executor durations, post-success no-ops, padding/validity masks and chronological
raw teacher/action correspondence are qualified. A future instrumentation fix
needs a fresh reviewed source/output identity and numerical proof that native
policy/loss/reset behavior is unchanged. No active source or mask is altered.

[Trace/source hashes and literal metadata](CONTROL-ACCOUNTING.json).
