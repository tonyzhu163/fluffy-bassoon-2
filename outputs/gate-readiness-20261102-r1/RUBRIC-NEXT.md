# Continue rubric iteration; compare explicit binary completion

The human-selected next iteration is the [visible completed-state drawer rule](../robocasa-flora-rubric-failure-sample-20261102-r2/revision1_3.json): reserve Score5 until completion is visibly established, retaining1–4 for progress. Its14-video graded check passes Linux preparation; after the earlier service failure, the human-requested retry is now running as Baige job-of3hcwyigjtu, with no completed result yet; see [status](../robocasa-flora-rubric-failure-sample-20261102-r2/STATUS.md).

The subsequent [twenty-video failure sample](../robocasa-flora-rubric-failure-sample-20261102-r1/RESULT.md)
has now tested two iterations of revision1 and matched direct-binary outputs.
The best graded wording reduces false successes12→7 on fourteen selected
failures; binary and numerical-predicate versions perform worse. The exact
draft below was not run verbatim and is not qualified. Continue from that
sample's case-level findings rather than interpreting this proposal as a result.

Human correction,2026-11-02: the central work is to encode the simulator's
task semantics accurately and keep iterating. Failure of revision1 and of
terminal thresholding does not end wording work. Binary labels are a proposed
output-format comparison, alongside the same improved criteria.

Source of the definitions below: pinned RoboCasa3443cdf3 and its inherited
helpers, already extracted with source hashes in
[NATIVE-CRITERIA-EVIDENCE.json](../robocasa-flora-rubric-audit-20261102-r1/NATIVE-CRITERIA-EVIDENCE.json).
These candidates are UNTESTED. Static task definitions are allowed in the
prompt; per-episode simulator joints, poses and outcome labels are not.

## Shared semantic instructions

```text
Identify the object, drawer, burner or dispenser requested by the native task
instruction. Keep that physical target fixed across all camera views.

Evaluate the state at the LAST visible frame using only the success conditions
listed for this task. Every listed condition must hold at the same time.
An earlier completed state that has been undone does not make this endpoint
successful. Do not combine conditions achieved at different times.

Do not add requirements about hand retreat, closure, orientation, stability,
appliance power or later actions unless this task explicitly lists them.
Judge the visible state rather than inferring completion from the robot's
motion or effort. Partial progress is not completion.
```

For a prefix, LAST means its actual end, with no future frames. First success
is the first prefix judged complete, at the available temporal resolution.
A multi-frame hold is a separate diagnostic with an explicit detection delay;
the simulator's one-frame first success is the reference.

## Task-specific clauses for the next candidate

**OpenDrawer**

```text
Identify the upper target drawer on the side named relative to the robot's
starting position, rather than camera-image left or right. The drawer has
one slide joint. The simulator requires outward displacement divided by
0.55 times the drawer fixture depth to be at least0.95; this is not a check
that every drawer in a stack is open. A visible gap or exposed interior
does not by itself establish that the target has reached this opening region.
The robot may still be touching the drawer: releasing it or moving the hand
away is not required.
```

**PickPlaceDrawerToCounter**

```text
The requested object must be in physical contact with a counter fixture,
and the acting gripper must be clearly away from the object (the native
distance condition is greater than25cm). Any counter fixture is accepted;
do not require a particular counter patch. Removing the object from the
drawer without reaching these two conditions is incomplete. Drawer closure,
a particular object orientation and an extra stability duration are not
additional success conditions.
```

**CoffeeSetupMug**

```text
The requested mug must be at the coffee dispenser's receiving position,
with less than4cm horizontal offset and less than10cm vertical offset from
the dispenser placement site. The acting gripper must be clearly away from
the mug (greater than25cm). Do not add requirements that the machine is on,
coffee is brewing, the mug touches the tray, the mug is upright, or the state
has lasted for an extra duration: those are not separate native checks.
```

**TurnOffStove**

```text
The burner named by the instruction must be off. Follow its corresponding
knob, even if cookware is on a different burner. The native off region is
around the knob's off position: its angle lies outside the on interval
0.35 <= abs(angle) <=2*pi-0.35 radians. Do not judge an unrelated burner or
require all burners off. The hand does not need to retreat from this knob.
```

The numeric clauses define the actual boundary; they do not establish that a
video model can resolve it. Read out near-boundary errors separately. A second
wording may translate a condition into visible language, but its mapping to
the same predicate must remain explicit. Do not silently replace contact with
stable placement or a distance condition with merely releasing the object.

## Output ablation, keeping the criteria identical

1. **Original control:** official1–5 prompt and unchanged expected-grade map.
2. **Native-criteria graded:** shared/task text above,1–5 output. Grade5 means
   that every native condition holds;1–4 retain partial-progress distinctions.
3. **Native-criteria binary:** the identical shared/task text, followed by:

```text
Return Success:1 only if every listed success condition holds at the final
visible frame. Return Success:0 if one or more conditions do not hold.
Do not award partial credit. Output exactly Success:0 or Success:1.
```

Binary prompting needs a separate system/output template and a parser/grammar
that permits0/1. Appending this sentence to a system still demanding1–5 is
invalid. Asking for binary labels is also different from thresholding a saved
five-grade score. If0/1 token probabilities are saved, bind their exact
extraction; do not treat the expected ordinal grade as P(success).

Human correction: begin with a small failure-focused sample from the634
development clips, plus a few true-success controls; do not replay the full
bank. The [20-video sample](../robocasa-flora-rubric-failure-sample-20261102-r1/PLAN.md)
iterates the first tested rubric and includes its newly introduced errors.
Keep video bytes, views and frame indices fixed across arms; keep original
and revised-system identities distinct. Compare paired false-success
corrections, new false successes, true-success losses, precision, recall and
within-case ranking. Categorize errors as target confusion, incomplete native
predicate, incorrect extra requirement, undone completion or visually
unresolvable boundary. Change wording based on those errors and repeat.

Freeze a candidate before independent validation on unused cases at the
intended horizon. The oldH750 videos can guide wording; they do not qualify
theH400 first-success reward arm or theH600 terminal arm. Production reward,
old caches and active runs stay under their existing contracts until a new
candidate actually passes. No binary or new-rubric replay has run yet.
