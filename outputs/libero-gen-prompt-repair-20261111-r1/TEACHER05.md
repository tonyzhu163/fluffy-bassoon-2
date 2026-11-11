# Persistent reference correction

Teacher04 exactly reproduced the successful 133-control grasp/lift prefix,
then exhausted its 80-control alignment budget without descending. Bowl–plate
XY separation decreased from 179 mm to 99 mm, and the wrist ended 23.3 mm below
its high-Z target. All seven joints remained inside their XML limits. The
full failure trajectory is preserved.

The servo reset its OSC position goal to the current wrist position plus at
most 5 mm each control. This bounded the instantaneous position error; it did
not implement a reference moving at 0.10 m/s. The earlier speed descriptions
in TEACHER04.md were intended reference speeds, not measured wrist speeds.

Teacher05 changes only that implementation. Initialize a persistent position
reference at the actual controller wrist pose after the same 133 controls.
Move this reference toward the bowl-corrected absolute wrist target with the
existing per-control bounds: 5 mm during high alignment, then 2.5 mm XY and
1.5 mm Z during descent. Pass the full reference-minus-current error through
the controller's unchanged input/output scaling and native action bounds.
Record both the reference and the absolute desired target, together with all
actual actions and observations.

The scene, seed, prefix, actual held orientation, 80/160 phase budgets,
5 mm alignment tolerances, 6 cm lift guard, rotation correction bound and
native goal stopping remain unchanged. These define a bounded diagnosis of
the servo correction. They do not establish a model score requirement.
One episode, one qualified RPBZZZ6 GPU, 10-minute scheduler limit and
420-second process timeout; no BPP loaded. If native execution succeeds,
candidate0's repaired prompt source becomes teacher05. Candidate1 retains
teacher03's deterministic first-native-success prefix.
