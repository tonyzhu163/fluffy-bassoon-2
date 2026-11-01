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
