# Evidence-led OpenDrawer rubric iteration

Active human goal: fix OpenDrawer reward through continued rubric iteration.
The previous goal turn made progress: completed28 paired labels show that
explicit visible completion fails to repair the error sample (7→8 false
successes, unchanged3/4 controls). Repeating stronger adjectives is not the
next justified intervention.

First inspect retained executed actions, native configuration, target fixture
type and joint-state helpers on allocated Linux CPU. Inspect existing videos
at their original resolution. Determine whether a faithful CPU-only replay
can recover the failed predicate for at most six retained trajectories, with
no motor inference or newly sampled actions. Validate replay identity against
retained native metadata, observations and terminal labels wherever available;
do not call a seed-only reconstruction an exact replay without evidence.
No hidden simulator state or old outcome labels enter Flora's prompt.

Then use at most six diagnostic rationale calls to understand the model's
visible interpretation, treating rationales as untrusted explanations rather
than simulator truth. Freeze next candidate(s) around the diagnosed missing
predicate and test on the same small drawer panel, retaining success controls
and a verbatim baseline. Keep first-success semantics separate from terminal
classification and report all regressions, not only corrected failures.

Do not rescore the full634-video bank. Each inference batch stays within one
96GiB GPU and15minutes/900GPU-s unless the human approves a materially larger
budget. No learner update or production reward change before qualification.
New-run contracts and evidence live here or in new uniquely named sample
folders; existing scientific outputs/worktrees remain immutable.
