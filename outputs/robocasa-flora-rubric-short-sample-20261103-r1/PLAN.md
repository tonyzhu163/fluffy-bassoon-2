# Short direct task instruction: six-video diagnostic sample

Freeze four graded arms on six existing OpenDrawer videos: the two physically
diagnosed failures112 and64, plus all four earlier success controls128,72,264,
144. The fixed sample deliberately challenges target identity and endpoint
semantics; it cannot estimate population accuracy. Do not run634 videos.

Control: verbatim revision1.1, including its original native instruction and
completion-criteria formatting. Candidates1.6/1.7/1.8 replace the long task text
with short native-aligned instructions. They emphasize respectively the
uppermost physical drawer, final-state success, and two explicit conjunctive
checks. Side is extracted only from the existing native language instruction;
no scene name, object identity, outcome, joint value or timing enters a prompt.
Static task geometry uses the source-correct about52%-of-depth threshold.

Official1–5 output template, model,64 sampled frames, three synchronized views,
and expected-grade map remain unchanged.24 labels total. Each candidate must
correct at least one of112/64 without losing any baseline accepted success
control to justify another small panel. Count all regressions and record
whether the previously rejected control72 is recovered. No candidate is a
production reward repair on this six-video evidence alone.

One96GiB GPU,32CPU/128GiB RAM/16GiB SHM, max15minutes/900GPU-s; no new robot
trajectories, rationale calls or PPO. Run Linux CPU hash/prompt validation first,
review current main, deploy through Git with exact source pin, and verify the
actual allocation GPU probe. Preserve existing runs/worktrees immutable.
