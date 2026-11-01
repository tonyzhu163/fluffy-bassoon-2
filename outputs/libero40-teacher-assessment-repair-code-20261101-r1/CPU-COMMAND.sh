# ONLY in the already approved new120-worker-second0GPU2CPU16GiB software gate.
# One shared deadline covers attestation, these30 focused tests, tiny dryrun,
# failures and cleanup. No59-suite repeat, model/actor/VLA/JAX/simulator loads.
"${REPAIR_CPU_PYTHON:?qualified CPU Python3.11}" -B -m pytest -q \
  tests/test_vla_source40_teacher_repair.py \
  tests/test_vla_source40_teacher_assessment_runtime.py::test_exact_public_pair_refusal_happens_before_next_steering_action \
  tests/test_vla_source40_teacher_assessment_runtime.py::test_actual_feature_value_drift_fails_even_when_all_hash_labels_match \
  tests/test_vla_source40_teacher_assessment_runtime.py::test_closed_resume_refuses_fresh_sibling_clock_before_server_or_reset \
  tests/test_vla_source40_teacher_assessment_runtime.py::test_closed_resume_does_not_replay_controls_or_qualification \
  --junitxml="${REPAIR_CPU_JUNIT:?fresh JUnit path}"
