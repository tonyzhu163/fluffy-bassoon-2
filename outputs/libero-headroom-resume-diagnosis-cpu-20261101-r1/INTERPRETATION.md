The first SAC learning slice closed normally at301,700 updates. Its fresh
same-source R2 continuation passed the allocation probe, then rejected the
atomic cursor/model/training consistency guard before any new episode or
update. The original full checkpoint was preserved with matching bytes.
The failed allocation's189 GPU-seconds remain charged; no cold restart or
blind retry is admitted.

The guard combines four comparisons: saved versus cursor training metadata,
restored update count, replay length, and the hash of reserialized agent state.
Its current exception does not identify the false predicate. Contract,
checkpoint SHA and motor RNG restoration were accepted before this guard.
Static inspection finds JSON-compatible training metadata, but that does not
prove which runtime comparison failed.

One concrete competing explanation is raw Flax MsgPack mapping order: the
saved signature hashes serialized bytes, whereas restore builds a fresh
agent template. Equal leaves can serialize in a different insertion order.
Actual actor/optimizer/target-state mismatch or cursor/counter mismatch must
also remain possible until measured. Existing fixture tests use a fake frozen
signature and do not establish real Flax serialization-order invariance.

The next discriminating check is the bounded allocated-CPU inspector: report
all four predicates separately, bind the legacy signature to the stored agent
bytes, and compare restored canonical state leaves with exact dtype/shape/value
checks using the existing state_differences/structure_digest helpers. Include
Adam, target critics, temperature, counters, replay and sampler/global/private
RNG, with original input hashes unchanged. No VLA, simulator, physics or SGD
is needed. Only an actually proven serialization-only discrepancy can justify
a migration preserving the stored-byte binding plus exact state equality;
removing the guard is not a repair.


Allocated diagnosis completed: corrected-runtime job-5zuyjva8qi9h passed four
focused tests and the actual OfficialSAC CPU inspection. Training, update-count
and replay-count predicates pass; only the restored raw serialization hash
fails. The original stored agent bytes exactly match the cursor signature.
All canonical actor/critic/target/temperature/Adam leaves compare exactly,
with zero differences and the same f0f15e...7183b digest. Replay content,
15,579 rows, all five sampler/global/private/motor-key RNG comparisons and all
five original input hashes are unchanged. There are27 mapping-order changes.
This confirms a representation-sensitive signature rejection for this saved
checkpoint, rather than a learner-state mismatch.

The first CPU attempt stopped at an incorrect parent runtime binding before
tests or model construction, costing76 CPU-worker seconds; its Failed status
is retained. The qualified Python3.11 retry costs83, total159/300, with141
unused closed. No GPU, VLA, simulator, SGD or robot action was used by this
diagnosis. A separate focused180-CPU-worker-second repair qualification is
funded; neither its code nor a new GPU continuation is admitted by diagnosis
alone. The stored-byte binding and exact state checks remain required.
