# Independent geometry review

Reviewed geometry02's full 16-scene contact sheet and raw per-layout JSON
records on November 11, 2026, before motor rollout. The geometry result
SHA256 is `0052a12ab3af51806a099919d0051b99bc05846a60ae76de7084fb80091a9f77`.

Both bowls are visible in every scene, with projected body-center markers
aligned to their bases. The visible left/up ordering matches the labels:
quadrants 0 and 3 are agreement; quadrants 1 and 2 are disagreement. The
bowls, plate and distractors are visually separated. The cabinet and stove
remain fixed. This review admits the scenes for motor testing; it does not
establish grasp or placement feasibility.

Independent reading of all 16 records confirmed:

- Sixteen distinct initial state hashes and no geometry errors or contacts.
- Eight layouts per stratum. Each rule selects each motor candidate four
  times and each physical bowl ID four times within each stratum.
- Both native predicates initially false. Positive placement diagnostics
  return `[1, 0]` and `[0, 1]`; all native sidecar parity checks pass.
- Minimum projected axis separation 8.8772 pixels, above the 6-pixel gate.
- Maximum object XY settling drift 0.03785 mm, below the 5-mm gate.
- Maximum holding-action end-effector drift 0.10521 mm and 0.04270 degrees,
  below the declared 20-mm and 10-degree diagnostic limits.

Geometry01 remains preserved as failed development evidence. Its episode
counter was not reset by restoring flattened physics. It also exposed
bowl/plate and bowl/cabinet contacts. The subsequent revision resets native
episode/controller state, matches the released runner's holding-action
orientation, and changes declared placements. Static fixture review caught
the stove burner's local +0.15-m X offset before geometry02. No policy
outcome informed these revisions.

The smoke uses layout indices 0, 5, 8 and 13 with both unchanged
motor prompts. The original proposal then called for the full 16-layout,
32-episode panel. This
pre-launch revision of the initial 0/4/8/12 proposal includes a swapped-ID
layout in each stratum while retaining one scene per quadrant. Repeated
smoke cases must not be pooled as independent confirmation. The distinct
execution count uses opposite one-hot native outcome vectors for the
paired motor executions; achieving both goals cannot count as an
informative alternative. All runtime-valid motor failures remain in the
panel, without tuning layouts or prompts on their outcomes.

After smoke01, the user rejected unjustified fixed pass rules and tests.
The previously proposed 7/8 cutoff is withdrawn, without changing the
completed geometry or smoke evidence. Report descriptive counts; inspect
the eight zero-success trajectories before deciding whether any further
panel would answer a useful question. The full panel is no longer automatic.
