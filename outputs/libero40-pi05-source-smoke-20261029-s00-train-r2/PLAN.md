# Source40 persistent-SAC runtime qualification

- Baige job-f0gv7b32ers5; exact source b48b3da95021d369f2cc9ede7cec8b5d07db7a83.
- One RTX6000 /16 CPU /96GiB RAM /8GiB shared memory;40-minute allocation.
- Official OpenPI pi05_libero40-task checkpoint, receipt ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170.
- Source task libero_spatial:00, training partition. Fixed state/seed within each4-episode case. SAC/replay persists across cases; training states0..39, separate validation states40..49.
- Target600 actual queried chunks; finish current case; hard ceiling7600 source primitives. Qualification/settling counted separately. This smoke does not visit all64 scheduled cases or constitute the full40-task corpus.
- Gaussian warmup until the first post-episode SAC update with replay>500; batch256, UTD20, fixed behavior throughout an episode.
- Full SAC/replay/optimizer/sampler/Python/NumPy/motor-key recovery every complete episode. Write-ahead intents conservatively charge ambiguous execution on recovery. Raw warmup retained; no non-learning prefix called AD.
- Qualification: actual allocated GPU probe;133 focused tests passed and2 JAX tests skipped in the probe environment. Actual OpenPI runtime independently passed both typed/legacy PRNG serialization round-trips.
- First observed progress:8 complete episodes,2 cases,76 query chunks,724 source primitives and90 settling actions; updates0 during warmup; full recovery37.9MB.
- r1 job-4bvfd79e25ym/source4f48888 exited1 after4m29s before source rollout, because a typed PRNG key cannot be converted directly to NumPy. Corrected via public JAX key_data/wrap_key_data; r1 artifacts and worktree preserved.
- No actor-expert, teacher-improvement, robotic adaptation or memory-benefit claim from this qualification. No90/Plus/Pro source data or semantic filtering.
- Full collection needs explicit case coverage, independent validation and a concrete interaction/compute/storage estimate; its larger budget has not been launched.
