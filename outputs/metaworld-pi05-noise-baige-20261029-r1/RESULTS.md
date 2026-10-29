# Meta-World released-motor screen: failed initialization

Baige `job-amup9a8mp0ym`, source
`4d4f46f4cdcd8cfd7c7056e95bf85d0a90e0ed59`. Scheduler terminal failure at
2026-10-29 18:28:05 Asia/Shanghai after 2 minutes 10 seconds running; driver
exit code 1. No simulator collection episodes or learning results were produced.

Both allocated GPU probes passed. The nine focused Linux checks passed with
zero skips, including real Meta-World goal placement and rendering. The
checkpoint matched the pinned Hugging Face hash. Model construction then
failed the installed OpenPI `transformers_replace` compatibility check:
the shared RLinf runtime supplies Transformers 4.55.4, whereas this standalone
evaluator's installed OpenPI class expects its replacement files with 4.53.2.
Successful imports and simulator tests did not qualify model construction.

After scheduler termination, uv installed Transformers 4.53.2 in the current
project's retained Meta-World environment, using independent file copies.
Five replacement Python files from the installed OpenPI package were copied
into that local Transformers package. The compatibility check passed, eight
CPU tests passed, and the protected shared Transformers initializer hash
remained unchanged. [runtime-repair.json](runtime-repair.json) records the
patch hashes and locations. The original shared package was not modified.

The retry has a fresh identity,
[`metaworld-pi05-noise-baige-20261029-r2`](../metaworld-pi05-noise-baige-20261029-r2/README.md),
with the same scientific source and checkpoint. Five GPU-minutes are
conservatively charged to this attempt; the retry is capped at 175 minutes,
keeping the combined ceiling at three GPU-hours. Historical source worktrees
and this failed run's artifacts remain preserved. The runtime repair is
explicitly recorded; this run's original package selection can be reconstructed
from the pre-cleanup inventory and shared-base paths.
