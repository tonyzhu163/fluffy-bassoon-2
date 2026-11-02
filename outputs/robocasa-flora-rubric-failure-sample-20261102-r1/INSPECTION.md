# What this iteration changes

Revision1 is copied byte-for-byte from the first scored revision. Revision1.1
adds robot-relative drawer targeting, the upper-drawer identity, a stronger
opening-distance distinction and explicit permission to keep holding a handle.
Stove text requires tracking the named burner's knob into its off region, not
inferring off from motion, cookware or an absent flame. Revision1.2 states the
same native predicates as a checklist with their static numerical boundaries.
Each new wording has graded and consistent binary-system versions.

The twenty reset/end diagnostic tiles were inspected before freezing these
wordings. Some failed drawers visibly open a long way; several endpoints hide
the drawer behind the shifted wrist camera or the cabinet face. These images
do not identify the exact failed joint predicate. Do not relabel them as
proven wrong-target or short-pull failures without simulator joint values.

Pinned native source in SOURCE-AUDIT.json shows TOP_DRAWER targeting and one
slide joint: normalized outward distance is displacement/(0.55*fixture depth),
with success at >=0.95. Left/right comes from the robot placement relative to
that fixture. It does not require every drawer in a stack to open. The prior
task audit supplies the stove interval: on iff absolute knob rotation lies
in [0.35,2*pi-0.35]. These are public task definitions, never an episode's
hidden state or label. None of the scorer inputs contains error categories,
truth labels, native seeds or cached grades.

The selection deliberately stresses revision1 errors: twelve of fourteen
failures were false full successes under that revision; two were corrected.
Four of six positive controls retained Score5 and two were wrongly downgraded.
Fresh revision1 is rescored to expose numerical or sampling drift separately.
No population precision, AUROC or first-success timing claim follows from
this terminal-video development sample. Binary outputs are hard labels, not
calibrated probabilities. All videos retain the official64-frame sampler.
