# Two external tests; original-LIBERO training corpus (v2)

- **Primary external evaluations: LIBERO-Plus and LIBERO-Pro**, reported
  separately. Train the attachment on original LIBERO, not their perturbed
  trajectories. Begin with the declared development cases below; this does
  not authorize collecting a full 130-task teacher bank.
- Target frozen backbone: `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`, revision
  `6222623f635769bfc73c9472e29fab9b7fd8e027`. Model weights/normalization and
  receipt are pinned in [protocol.json](protocol.json). Original-130 training
  coverage is disclosed by the release; it is not a held-out-VLA-task claim.
- The previous [v1 split](../libero-pi05-corpus-20261029-r1/split.json) stays
  immutable: 78 original training / 10 validation / 10 official internal
  held-out tasks + two excluded full-goal duplicates. Retain that controller-only
  task-transfer test as auxiliary. No relabeling of its existing data or runs.
- Original-LIBERO source teachers, student gradients and normalization remain
  confined to the declared training bank. Validation-only original tasks
  guide model selection. No final Plus/Pro case may supply teacher collection,
  meta-gradients, normalization fitting, curriculum or checkpoint selection.
- Switching from older OpenPI weights to RLinf130 changes the motor and
  feature coordinate contract. Regenerate compatible source data; do not
  silently reuse old teachers or call their results RLinf130 results.

## Immediate development cases — fixed before outcomes

- Base: original `libero_object:1`, cream cheese → basket. Exact typed full
  goal matches training family `libero_90:47`. This is a development base
  alias, not an expansion to all 130 source teachers.
- `libero_90:57` is cream cheese → tray and has no perturbed Plus counterpart
  in the staged index. It must not be relabeled as a Plus test case.
- Plus: runtime index **667** / classification ID **668**, camera scale 115,
  robot init 0; runtime index **290** / classification ID **291**, camera scale
  100, robot init 2. Official names and hashes are in the manifest.
- Pro: `pro_object_swap_init0` and `pro_object_appearance_init0`, from the same
  base. Source, HF asset revision, exact BDDL/init hashes are in the manifest.
- Development initial state 0; reserved development RNG seeds 0 and 1000;
  actual seed use recorded in each run. Start K=4 baseline/ordinary-DSRL
  adaptation probes. No trained meta-controller result is implied.
- Plus development horizon **280**, settling **10**, from pinned RLinf's
  standalone original-LIBERO object evaluator. Pro development horizon **280**,
  settling **5**, from its pinned render utility. Preserve each basis; these
  small development jobs are not full official leaderboard reproductions.
- Fresh environments and seeded resets; preserve actual initial-state vectors,
  observations, actions and hashes. Pixel equality is a measurement, not an
  indefinite prerequisite or a reason to replace benchmark dynamics.

## Final tests and reporting

- Freeze exact final case inventories from each pinned official index before
  looking at final outcomes. Include source/asset hashes, canonical base and
  typed-goal family, perturbation definition/parameters, initial state and seed.
  Do not invent case IDs or infer original IDs from expanded task positions.
- Keep the entire declared development base/goal family out of blind final
  case inventories; reserve final seeds ≥100000. No development condition,
  parameter/seed instance or its sibling family may appear as an unseen final
  case. Parent manifests must make those exclusions explicit.
- The staged Plus classification uses **one-based IDs**, while runtime order-0
  indices are zero-based: all 10030 row names were statically verified at
  `map[classification_id - 1]`. `libero_90` contains 90 unperturbed tasks and
  is not the Plus perturbation bank.
- Separate external strata: perturbations of attachment-training base families
  measure perturbation transfer; perturbations of held-out attachment bases
  additionally measure base-task transfer. The frozen VLA may know every
  original base task. Report these distinctions, not a single unseen-task claim.
- Report official-style **first-attempt** success using documented suite
  settings, coverage and denominator separately from **K=1/2/4/8 adaptation**
  trials. If a full official catalog includes development cases, disclose that
  overlap and also report the blind development-excluded score; do not present
  a development-contaminated aggregate as an untouched final test.
- Adaptation: preserve the declared perturbation within a trial; use only
  earlier actual feedback. Freeze student/motor parameters; ordinary DSRL
  comparator may update its steering policy. Count every attempt, primitive
  action and settling action; retain first/final/success-once curves and source
  cost. Numerical reward, visual feedback, static/reset and simple-search
  controls remain distinct questions.
- Final case expansion and official aggregate coverage are pending the pinned
  runnable inventories; this does not block the already declared parallel
  Plus/Pro development runs. No extra scientific gate is introduced here.

Official sources: [Plus](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182),
[Pro](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec).
