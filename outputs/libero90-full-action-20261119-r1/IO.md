# Full-action LIBERO-90 rollout interface, version 1

The manifest is the sister run's `../libero90-meta-retry-20261119-r1/cases.json`. Paths below are relative to the episode root given to the runner. A published trial contains `trial.json` with its exact `manifest_row`, source pin, reset fingerprints, episode costs and native outcomes. Writers publish the entire directory by atomic rename. Readers consume only published trials.

    fitting/task00-r00/{support,query0,query1}/
    development/taskNN-rNN/{support,query0,query1}/
    live/{development,test}/taskNN-rNN/{support,base,matched,meta_intact,meta_cleared,imitation_intact,imitation_cleared}/

Every episode has `public.npz` and `result.json`. Load archives with `allow_pickle=False`. The public tape has chronological issued actions `issued[T,7]`, post-control wrist `positions[T,3]`, pre-first-control `initial_position[3]`, post-control jaw `gripper[T,2]` and native `rewards[T]`. The scalar `controls` is T. The scalar boolean `terminal` is native terminal success. A horizon failure retains all 400 controls, has terminal false and is marked truncated in result.json. The JSON fields `controls`, `score` (sum of native rewards), `success`, `truncated`, `policy_calls`, `feature_calls` and timings describe the completed episode. Settling controls are separate.

Each query also has `decisions.npz`. Its arrays are finite float32 `state[J,8]` (position, axis-angle, two jaw positions), `embedding[J,D]` (pooled frozen visual/language prefix, expected D=2048), native-precision `proposals[J,K,10,7]`, float32 `scores[J,K]` (before execution), integer `index[J]`, `offset[J]` (zero-based physical control count at decision) and `length[J]` (executed prefix length, 1–5). Concatenate state then embedding to recover the frozen query feature. K=2 throughout except for live `base`, where K=1. Both collection branches choose by independent fair coins, with prospective constant zero scores. The native result labels only the chosen trajectory. All failed episodes are retained. Streams are keyed by manifest seed, arm and proposal slot, with a separate coin channel. The current collector also records the first noise hash per slot in `result.json/opening_noise_sha256`. Independent draws that happen to yield identical commands are retained.

## Frozen reader import

The collector starts a persistent CPU-only Python subprocess in the reader environment. It imports the sibling `model.py`. That module exports this interface for each frozen recipe.

    scorer = load_scorer(path: pathlib.Path, recipe: str)
    scores = scorer(support, state, embedding, proposals)

`path` is a single self-contained portable weights file for the recipe, including ensemble and normalization parameters. Its SHA-256 is recorded. `recipe` is exactly `meta` or `imitation`. `support` is a dictionary of the seven numeric public tape arrays named above, or Python `None` for cleared history. The reader constructs its history including the episode boundary from this tape. `state` is float32 [8], `embedding` float32 [D] and `proposals` float32 [2,10,7]. The callable returns two finite scores in proposal order as a NumPy-compatible array, averaging logits across frozen seeds before returning. Calls are stateless with respect to query history. All tensor/device conversion belongs in the reader. The collector executes the first argmax. It supplies no query rewards, earlier query decisions, timing, IDs, seeds, sibling outcomes or evaluator metadata.

Across live policies, proposal noise is shared by trial, replan ordinal and candidate index. Each policy resets the simulator and visits its own observations. Base executes one draw. Matched computes both and executes candidate zero. For live matched and learned queries the collector also makes one feature call per decision. Base runs one native action forward and zero auxiliary feature calls. For its archive only, it reuses the matched arm’s diagnostic embedding after verifying the exact current motor-input images, state and prompt at that ordinal. Both tapes include these input hashes in `input_sha256[J]`. The runner also verifies identical complete issued, wrist and reward traces. Support uses one native draw per five controls.

## Running on Baige

From the clean deployed worktree, set `pin` to its verified full Git SHA. Run smoke on the existing development GPU. It selects only the first registered fitting row and publishes three episodes. The separate `smoke-live` entry point qualifies the reader bridge on the second fitting row. Full panels require a one-GPU Distributed Training allocation. The runner writes the GPU probe, run log and exit times under `EPISODE_ROOT/.attempts/`. Use the same episode root, pin, weights and shard arguments to resume interrupted work. A complete trial is verified and skipped. An incomplete trial is repeated from reset.

    run=outputs/libero90-full-action-20261119-r1
    pin=$(git rev-parse HEAD)
    out=/mowenjie/fluffy-bassoon-2/$run
    bash "$run/runner.sh" "$pin" smoke "$out/smoke01"
    bash "$run/runner.sh" "$pin" fitting "$out/episodes" --shard-index 0 --shards 1
    bash "$run/runner.sh" "$pin" development "$out/episodes" --shard-index 0 --shards 1
    bash "$run/runner.sh" "$pin" live-development "$out/episodes" --meta "$meta" --imitation "$imitation"
    bash "$run/runner.sh" "$pin" live-test "$out/episodes" --meta "$meta" --imitation "$imitation"

Set `meta` and `imitation` to the frozen weight files produced by the model worker. For parallel collection use a common shard count with distinct zero-based indices per allocation. The stages above describe separate allocation entry points.

## Measured qualification

The 2026-11-19 Baige development RTX 4090 smoke used this exact command and exited 0. The trial, design, array hashes, GPU probe and exit receipt are under the shared `sourcecheck01` output.

    root=/mowenjie/fluffy-bassoon-2
    run=outputs/libero90-full-action-20261119-r1
    pin=d1b0eb85cc6186e269efff02eaeb28caedccafaa
    bash "$root/setups/libero90-full-action-20261119-r1/source-d1b0eb85/$run/runner.sh" "$pin" smoke "$root/$run/sourcecheck01"

| Fitting task 0, repeat 0 | Pose | Native success | Controls | Action forwards | Feature calls |
| --- | ---: | --- | ---: | ---: | ---: |
| Practice | 14 | No | 400 | 80 | 0 |
| Query 0 | 15 | Yes | 294 | 118 | 59 |
| Query 1 | 15 | No | 400 | 160 | 80 |

The three episodes took 92.38 s together, including reset, first-call compilation and tape writes. The runner took 115 s including the probe and model startup after obtaining the lock. This fitting trial used 30 additional settling controls. The native-precision NPZ tapes occupy 1,328,637 bytes. The readback reproduced all 139 coin selections and exactly matched each physical command to its selected proposal. Both opening feature vectors matched, including the 2048 pooled values. Both branches had distinct candidates at every replan. The CPU learner read the published trial tapes successfully.

A separate fitting-pose live qualification at `57a01360` used six-update fixture readers fitted on the first case. Practice and all six own-policy retries reached the 400-control horizon on the fresh case. The seven episodes took 183.23 s, or 209 s including startup and the probe. They used 960 native action forwards, 400 diagnostic feature calls, and 1.81 s in the CPU bridge for 320 learned decisions. CPU readback reproduced all 640 learned logits, all four learned arms’ choices and all issued actions exactly. The matched and single-draw arms had identical current motor inputs, candidate-zero proposals and physical control tapes at all 80 replans. Base made zero auxiliary feature calls. Both qualifications used fitting poses exclusively.

For a rough 4090 budget estimate, replacing each cold practice’s duration with a measured warm 400-control single-draw rollout gives about 67 s per logging trial and 159 s per seven-episode live comparison. On these two fitting cases, extrapolation gives 17.9 GPUh for fitting, 2.7 for development logging, 6.3 for live development and 31.7 for live test, about 58.6 GPUh altogether. Episode lengths, accelerator type and per-shard startup can change this estimate. The frozen ensemble paths from a completed fit are `FIT_RUN/frozen/meta.pt` and `FIT_RUN/frozen/imitation.pt`.

The exact live qualification command used the same `root` and `run` variables as above.

    pin=57a01360b0d342b19ee39028c21402a512b115d8
    weights="$root/$run/nativefitcheck02/weights/frozen"
    bash "$root/setups/libero90-full-action-20261119-r1/source-57a01360/$run/runner.sh" "$pin" smoke-live "$root/$run/livecheck01" --meta "$weights/meta.pt" --imitation "$weights/imitation.pt"
