# Interim direct-action deployment qualification

All four frozen snapshots completed the actual native deployment qualification:
AD11900, DPT11400, DIT3800 and IC-IQL3800. These are interim validation-selected
candidates. Training continues under the original convergence rule; these
snapshots do not become final candidates.

Each method ran two cases on each of three ID tasks, six actual episodes per
case, in both full and episode-reset modes. The independent raw audit passed
all288episodes,172800physical controls,3456chunks,864videos and519264decoded
frames. All physical commands, initial pairings, feedback and success receipts
were checked. Deployment used zero SGD.

The first raw audit incorrectly expected an already-cleared entry cache in
reset mode. Entry is recorded before act(), which clears the cache on the
first query of each episode. The corrected audit also requires exact
previous-exit/next-entry signature continuity. A separate CPU replay inspected
all144reset tokens: prior action/reward/done were exactly zero and the policy
received no cache. All3456saved GPU controller outputs agree with CPU replay
at relative/absolute tolerance1e-4; maximum absolute error is8.5354e-5.
The stricter1e-5 cross-device check failed and is retained as a precision
observation. This is numerical agreement, not byte-exact replay. Rollouts,
weights and training were unchanged.

The two-case counts in SNAPSHOT-RESULT.json are deployment diagnostics. They
are too small to rank methods or establish a benefit from previous episodes.
Final50-case evaluations still require each method's convergence receipt and
its validation-selected frozen checkpoint.
