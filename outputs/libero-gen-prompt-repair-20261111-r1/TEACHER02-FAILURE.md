# Teacher02 bootstrap failure

Scheduler job `job-8x3eo4uscgrd` was created at 20:34:33 Shanghai and
entered Running at 20:34:48. Its log at 20:34:49 reported Git 2.34.1 and
`unknown capability 'filter=blob:limit=102400'`. The failure preceded
checkout construction, the GPU probe, and every simulator episode. The
complete bootstrap clock remains on shared storage; final scheduler
accounting is pending while the development endpoint is unavailable.

The32,492-byte command used a standard v3 filtered Git bundle rather than
the initial534KB full-bundle command. Only the small command was submitted.
The missing object is the already retained teacher01 failure screenshot;
code, commit and tree objects are included. The next compatibility wrapper
uses the same unchanged PACK through Git's `index-pack --promisor` support,
verifies the target commit and required code, and excludes the screenshot
from the sparse working tree. It does not change the teacher recipe.

Git2.34.1 explicitly supports that import flag in its [official source](https://raw.githubusercontent.com/git/git/v2.34.1/builtin/index-pack.c).
The [bundle format](https://git-scm.com/docs/gitformat-bundle) specifies the
promisor marker for filtered packs. The wrapper implements those semantics
explicitly because the older bundle reader cannot parse that capability.
