# Initial optimizer raw signature differs; no gradient replay ran

Job`job-01rwmdh7muey` restored qual02's native headless step0 checkpoint.
All four frozen-VLM hashes match. Persisted rank1/2 comparisons show exact
trainable weights, scheduler, scaler, RNG and optimizer_steps0; only the raw
optimizer-state digest differs (expected4ed06e… versus actualdd4351…). This is
not proof of numeric optimizer-state loss or proof of harmless representation.
The original signature did not retain a structured state/group reference.

No new simulator controls, gradient replay or optimizer update occurred.
Do not weaken the exact recovery requirement or infer success from weights
alone. The separate metadata-only opt01 diagnostic records named parameter
ownership, state keys/shapes/dtypes/data hashes/scalars and all group settings
before/after native save and independent load. It claims no qualification and
does not advance training.

Created03:57:54Shanghai, released04:01:36. Scheduler209sec×4=836GPU-sec;
creation-to-release222sec×4=888; worker168sec×4=672. The later actor-handle cleanup
exception is secondary to the optimizer signature failure.
