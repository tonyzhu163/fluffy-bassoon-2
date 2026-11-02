# Paired reset images: diagnostic comparison

Human direction on 2026-11-02: the one-level RGB maximum was too strict;
fix the check and examine recovery of the interrupted Stage 0 screen.

## Change

Source `f804ce6a586d65a941bdb3d5e3f2e7d4344b5a18` makes RGB differences
diagnostic. A maximum-pixel difference no longer rejects an otherwise
identical native case. Every view retains maximum, mean and nonzero-component
differences, and the report explicitly says `image_comparison=diagnostic_only`.
The policy receives its original pixels without rounding or replacement.

Exact case, simulator-state/scene/camera fingerprints, public proprioception
and task instruction remain mandatory. Missing cameras, wrong image
shape/dtype and invalid public state still reject the pair. This is a revised
comparison policy for future sources, not a retroactive change to an active
or historical run's pinned contract.

## Qualification

On the existing Baige Linux development allocation, with CUDA hidden:

- All 28 affected pairing and rollout tests passed, including exact-state
  rejection, missing/invalid cameras, unchanged policy pixels, uint8
  subtraction and completed-lane checkpoint callbacks.
- The exact saved TurnOffStove reset that stopped the original screen was
  replayed against its retained canonical snapshot. Physical and public
  state match exactly; right-camera maximum delta is 2, mean delta is
  0.000396728515625. The new diagnostic comparison passes.
- No new simulator controls, model inference or PPO updates were performed.

Evidence: [unit qualification](QUALIFICATION.json),
[saved-reset replay and integration tests](SAVED-RESET-REPLAY.json).
macOS checks were limited to source syntax and whitespace.

## Recovery boundary

The original Stage 0 job `job-16bb8e4z2n3x` is terminal. Its source and
output must stay immutable. The existing screen owner (`01a19b58`) retains
the runner/checkpoint repair and launch ownership; do not start a competing
allocation from this repair.

Use a fresh source-pinned attempt/output and explicitly record this image
comparison migration. Reuse the original case bank, complete anchor records
and verified completed native-trial truth/video evidence. The interrupted
native8 banks lack per-chunk features/actions: their completed trials can
contribute verified success outcomes but cannot be called complete training
histories. Collect missing trials, retain all old costs and unresolved calls,
and checkpoint each newly completed episode before refilling its lane.
Check exact case/draw/truth identity before skipping any completed trial;
never silently replace or double-count it.

The owner's terminal audit recorded 345 complete native episodes,
zero alternative-interface episodes and zero PPO updates; conservative cost
2,924 GPU-seconds leaves 40,276 of the original 43,200 GPU-second screen
ceiling. This repair adds no training or resource grant. Resume is not yet
claimed: the owner must bind and qualify its fresh recovery entry first.
