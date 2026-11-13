# Task selection review before GPU launch

The user approved the additional230GPUh phase, then questioned whether the tasks were representative. No GPU allocation has been submitted. Budget authorization remains valid; the prepared Cabinet/Rack/Counter trio is not treated as a locked representative set.

The November12 plan added Cabinet, Rack and Counter-to-Stove to the then-confirmed Drawer/Toaster/Faucet teacher inventory. The November13 scaling proposal carried those additions into the different Drawer/Toaster/Coffee transformer corpus. That loses the control-manipulation category. Reusing started pilots reduced collection cost, but was not a systematic representative sampling argument. The earlier November3 portfolio had explicitly balanced open/close, pick-and-place and buttons/knobs; the current proposal should have retained that design discipline.

| Broad manipulation family | Fixed three-task corpus | Initially proposed additions | Better coverage candidate |
| --- | --- | --- | --- |
| Articulated opening/closing | OpenDrawer; CloseToasterOvenDoor | OpenCabinet; SlideDishwasherRack | Keep the two original tasks; avoid spending both new slots on this family |
| Object transport and placement | CoffeeSetupMug | PickPlaceCounterToStove | Retain Counter-to-Stove to add object/container variation |
| Small control manipulation | None | None | TurnOffStove and TurnOnSinkFaucet, subject to corrected-runtime and source-quality checks |

The recommended candidate six is Drawer, Toaster, Coffee, Counter-to-Stove, Stove and Faucet. This is a deliberate two-per-family manipulation pilot, not a random representative sample of all365 tasks. It still lacks navigation, broader contact/insertion behavior and long-horizon composites. Composites remain outside the current source-fitting scope. Preserve Fridge and Microwave as the shared historical held-out tasks; do not relabel old source/OOD panels. Faucet was held out in an older different split and later used for other learners, so any new Faucet inclusion must be explicit in this new corpus.

Selection should account for four axes:

1. Skill and fixture coverage, with within-family variation rather than many near-duplicate tasks.
2. Kitchen, object, geometry and initial-condition coverage, with complete train/validation/evaluation separation.
3. Difficulty and source learning dynamics: retain successes, failures, recovery and chronology, and report flat teachers instead of selecting only favorable outcomes. Do not impose an invented success-rate gate.
4. Information available for adaptation: inspect whether earlier actions and rewards can help the later query and whether source support/query histories match deployment. Diverse demonstrations alone do not establish meta-learning.

ID should use fresh cases from represented source tasks. OOD should distinguish new tasks/fixtures within represented families from entirely held-out skill families. Task holdout and unseen kitchen/object variation are separate axes. Maintain full-versus-cleared history comparisons; add targeted reward/donor interventions where justified within the existing scope and cap. Two held-out tasks give a narrow transfer test, not broad coverage of the benchmark.

The purpose of the three-day readout is a decision about this tested recipe. Failure on a small or uninformative source distribution is not evidence that the general meta-learning direction is impossible. A stronger negative requires a justified task distribution, usable source learning signal, and a sufficiently precise history-effect comparison.

The repository evidence check found an existing corrected-critic Faucet40 source. Its [first-update qualification](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-FAUCET-ADVANCEMENT-01.json) records zero initial rollout/recomputed value difference on both actors, and its [completed source readout](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-RESULT.md) reports 204 to 242 successes on 512 paired cases. This is a candidate for reuse rather than another automatic PPO extension. Current artifact availability, full chronological input mapping and query-label compatibility still need checking; the small November10 LoRA dataset is not the full transformer corpus.

The retained Stove130 source belongs to the older critic implementation and had flat source performance ([historical readout](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md)). It cannot be presented as a new corrected-PPO source. A fresh corrected-runtime pilot would be needed if Stove is selected. Replacing two continuations with one fresh pilot and a reused Faucet source may fit the existing collection allowance, but that cost estimate is not yet qualified.

Next work is to verify Faucet input availability, cost a fresh Stove pilot, and lock the coverage-based split before collection within the approved230GPUh total. Existing Cabinet/Rack pilot data and checkpoints stay preserved. The old continuation scripts remain a qualified fallback, not the approved final task-selection decision.

Sources: [original pilot selection](../robocasa-six-source-decision-20261112-r1/PLAN.md), [earlier balanced portfolio](../robocasa-flow-sde-10m-per-task-20261103-r1/TASK-PORTFOLIO.json), [RoboCasa atomic-task scope](https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html), and [RoboCasa365 task definitions and data-composition study](https://arxiv.org/html/2603.04356v1). The paper supports studying task/data composition; its behavior-cloning experiments are not proof of our frozen-model in-context learning hypothesis.
