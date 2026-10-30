# Full shifted-suite inventory preparation

Metadata only, no simulator/model/tests/GPU or source-pretraining data. All
registered cases in the pinned primary benchmark are retained; no semantic,
outcome or development-case exclusion was applied.

The pinned [Plus release](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md)
specifies one evaluation trial per variant. Its classification IDs and names
match the static task map exactly: Spatial2,402/Object2,518/Goal2,591/Long2,519,
total10,030 across seven categories. The original90 map is a separate entry,
not another shifted suite. With our explicitly declared OpenPI family horizons
220/280/300/520 and10 settling, one K0 method has a3,420,960-action upper bound;
five methods17,104,800. Five K0 methods plus four full/reset K1 groups per
variant require130,390 real episodes and44,472,480 maximum actions. These are
planning bounds, not a funded run or assertion of complete upstream seed/init
and runtime equivalence. Support costs cannot disappear into a query score.

The pinned [Pro evaluator](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/README.md)
declares five separate perturbation types across four suites. The task map
registers200 such tasks. At dataset revision
c86fc3b8293185a6f373677018ff3e37f8391602, all160 object/position/language/task
BDDL/init pairs are listed; the40 environment tasks need the official
environment-generation/staging path. The saved Hugging Face recursive tree
and complete siblings list agree on676 files/10,269,485 bytes. Additional
experimental registrations are not silently relabeled as the five-dimension
primary benchmark. No case was removed based on its instruction or outcome.

For Pro,68,000 actions bound one initialization repetition of200 tasks under
the declared family horizons/settling. Actual bank lengths, repeated nominal
states and the official evaluation repetition setting still need to be bound
on allocated Linux before quoting a concrete full-suite episode/GPU request.
A seed or initialization index is not proof of a new physical state.

The current working adapters admit only the old tiny development panels.
Neither full Plus nor full Pro is qualified or evaluated by this inventory.
This work is independent of source-SAC collection admission and must not delay
it. Full-suite target compute is outside the pending25h source request.

[inventory.json](inventory.json) records all counts, category breakdowns and
SHA256s of the static source metadata. Raw public metadata were copied read-only
from the existing pinned Baige sources into the ignored raw/ directory;
trajectory/state archives were not deserialized. Full Pro data staging is
separate and preserves the original two-pair development data root.
