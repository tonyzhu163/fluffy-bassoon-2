# Existing frozen-pi0.5 DSRL capture: read-only readiness

Scope: inspect existing `vla_libero`, `vla_source40_corpus` and target-online
helpers against the DSRL-first priority. No model, tests, collection, relabeling
or new framework was run. Native PPO and its25-step continuation are queued.

The existing path already retains compatible steering data because both
behavior and eventual noise teacher act through the **same frozen motor**.
This is different from the native action-expert PPO corpus.

| Needed later | Existing retained data |
| --- | --- |
| Raw VLA current/query and next observations | `vla_libero.run_episode:439–442`: `motor_observation`, `motor_next_observation`, instruction, both image/wrist views and proprio; canonical pixel64/proprio8 SAC observations also retained |
| Actual steering action | `executed_noise` is the32D coordinate actually sent after bounds; `native_initial_noise` records its exact H10 tiling; warmup remains unbounded Gaussian and learned policy uses clipped noise |
| Robot action and duration | `executed_actions`, raw primitive rewards, `primitive_steps`, requested chunk size; actual executed array length establishes the exact leading duration mask |
| Feedback/boundaries | Sparse steering reward, success/done, termination versus truncation, actual-duration discount |
| Chronology/identity | Ordered episode row list, case/reset IDs and scene seed in summary, full initial simulator/public hashes/state where provided, training/query episode ID and per-row source/target update clock |
| Producing actor identity | Target helper signs the actor/critic state before each episode, proves it fixed through collection and labels the subsequent update count separately (`target_online_dsrl:270–297`) |
| Interrupted prefixes/costs | `JournaledSimulator` persists raw RPC input/context before execution, complete returned observation/action/reward outcome before SGD, and conservatively charges unresolved intents |
| Complete recovery | `OfficialSAC.save/restore` includes actor, critic, target critic, temperature, optimizers, replay, NumPy sampler, Python/NumPy/JAX RNG; callers add typed motor RNG and source cursor after every episode |
| Final candidate teacher | Immutable milestone/source-publication full recoveries with hashes; source exporter calls final candidate `eval_actions` on each saved SAC observation, labeling only target noise and preserving behavior histories |

Explicit per-row chunk/time indices and prefix masks are not yet named fields
in legacy rows, but derive exactly from list order, cumulative durations and
the retained executed-action lengths. The actual after-bounds z is retained;
legacy `proposed_noise` can already be clipped by the choosing wrapper and must
not be advertised as the raw unbounded preclip neural output. No such naming
convenience is a prerequisite for measuring the ordinary DSRL baseline.

Prefix4104 features can be extracted later on allocated resources from saved
raw observations. Their optional immediate capture is not required for SAC or
for preserving a future AD/DPT corpus; it should not delay the first baseline.
The current actor input remains the canonical released pixel64/proprio8 path.

AD noise targets can be the actual executed z. DPT needs a distinct later
immutable label pass with a preselected final candidate SAC actor, exact
checkpoint/config and frozen-motor/noise-interface identities. A saved actor
is not automatically competent. If training/validation labels are intended to
measure heldout-state prediction of one teacher, use the same selected TRAIN
actor on both; legacy independent validation streams used a different actor
and must retain their honest teacher-transfer metric. No label generation is
a gate for the first ordinary DSRL run.

The concrete task57 route is not an existing ready CLI: current main entry
points hardcode0/1, six development tasks or0/40, and reusable `run_case` also
enforces one fixed nominal state across all training/query episodes. Author-
style varying scene training therefore needs a narrow population adapter with
saved sampler/case order; simply substituting task57 into the fixed-case loop
would misstate the experiment. The fidelity/readiness owner is reporting that
route/recipe to root; this inspection does not duplicate its implementation.
