# Native collection, fitting and evaluation

Use the current run's native.py. Set R2_CHECKOUT to the original fc61bf2a evaluator checkout, CROSSARM_SIM_PYTHON to its simulator interpreter, and use R2's release, simulator, CUDA and EGL environment. OUT is a fresh job result directory. PANEL is the original large-panel-v3 root. FIT is sft02/fit. WEIGHTS is the R2 PaliGemma base snapshot. Use one GPU process with eight independent simulator children, one arm at a time.

```sh
"$CROSSARM_SIM_PYTHON" native.py prepare --out "$OUT" --panel "$PANEL" --fit "$FIT"
"$VLM_PYTHON" native.py collect --out "$OUT" --panel "$PANEL" --fit "$FIT" --weights "$WEIGHTS" --workers 8
"$VLM_PYTHON" native.py fit --out "$OUT" --model "$OUT/fitted"
"$VLM_PYTHON" native.py evaluate --out "$OUT" --panel "$PANEL" --fit "$FIT" --weights "$WEIGHTS" --workers 8 --model "$OUT/fitted"
```

The collector and scorer use the exact sibling MODEL-API.md signatures. Each OUT/train/{Panda,Sawyer}/<seed>.npz contains split, arm, layout_id, layout_seed, raw_success, teacher_success, decision_reached, fork_x, suffix_x, base_chunks, teacher_actions and valid_lengths. Suffix rows include unsuccessful teachers. Invalid trailing actions are zero. For early termination the stored fork_x is empty and the fit loader supplies None. Fit enumerates the 128 expected files from the training manifest. No development trajectories enter the fitter.

Teacher API is teacher.make_teacher(arm_env, public_obs) -> callable(public_obs) returning one action per control. Only training children can invoke the teacher. Score is latched once at 60. Always and gated apply the same fitted policy and caps. Evaluation writes OUT/eval/{Panda,Sawyer}/<seed>.json plus its three-branch native NPZ, then OUT/evaluation.json. Per-seed records include all three native outcomes, the decision, and archived raw layout/outcome comparisons. NATIVE-READY marks implementation handoff to the parent for pinning. Ops owns qualification and launch.

For the Linux pilot run prepare first, then run the following single process before full collect. It reads the first two admitted training rows. It collects one training pair per arm on the first geometry, fits into OUT/pilot/fitted, and executes all three controllers per arm on the second geometry. All pilot artifacts live in OUT/pilot. The full collection independently executes the 64 registered training rows per arm and the production fit reads only OUT/train. Pilot evaluation uses reserved training data only. The pilot uses the same model, teacher and runner APIs and the same per-branch checks.

```sh
"$VLM_PYTHON" native.py smoke --out "$OUT" --panel "$PANEL" --fit "$FIT" --weights "$WEIGHTS" --workers 8
```

Independent readback must use the final per-branch NPZ namespace `trace_{mode}_{field}` for actions, base, native_steps, rewards, success, terminated, truncated, observations, and reset. The unprefixed `teacher_actions` is the padded suffix target and `teacher_success` the scalar outcome. The archived raw opening comparison is named `opening_observation_match` and covers both cameras, state and prompt. The earlier draft of evidence/verify.py used the pre-final key names.
