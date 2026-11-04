# Early native deployment qualification while exact training continues

Freeze one current validation-best candidate per method, then run only the
existing two-case/task ID qualification bank. Both full memory and episode reset
execute six actual H600 episodes. Four independent1GPU replicas preserve the
fixed B2 cohort per method/mode, qualification seeds and common native draws.
Maximum45minutes/replica=3GPU-hours inside the existing96GPU-hour ceiling.

This moves the deployment check ahead of optimizer convergence, so motor,
normalization, history and native artifacts can be checked before the large
panels. The diagnostic snapshots never become final candidates by this route.
Qualification results do not alter training data, stopping rules or checkpoint
selection, and are not used to claim method superiority or adaptation. Final
fifty-case panels still require each method's declared validation convergence
and its then-selected checkpoint. The qualification and final case banks remain
disjoint. Current direct-action heads remain distinct from learned steering
through the frozen native action expert.

FREEZE-SNAPSHOTS.py checks the actual old source/corpus/model contracts, copies
each actor with before/copy/after hashes, preserves any raced copy, and records
the immutable candidate manifest and observed training status. READOUT permits
this explicit snapshot override only for ID/qualification/count2/shards1; the
final gate is unchanged. Active learners keep their original worktree, Adam,
sampler, RNG, plateau counters and full recovery checkpoints.
