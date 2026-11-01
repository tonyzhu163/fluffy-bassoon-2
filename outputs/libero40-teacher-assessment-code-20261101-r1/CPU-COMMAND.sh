# Run only on the existing new180-worker-second0GPU2CPU16GiB allocation.
# Existing sole owner surrounds this with actual zero-GPU proof, inclusive clock,
# exact Git/import manifest, deadline/teardown and scheduler cost recording.
# No package install/environment mutation or old suite rerun.
"${ASSESSMENT_CPU_PYTHON:?qualified CPU Python}" -B -m pytest -q \
  tests/test_vla_source40_teacher_assessment_profile.py \
  tests/test_vla_source40_teacher_assessment_runtime.py \
  --junitxml="${ASSESSMENT_CPU_JUNIT:?fresh actual artifact path}"
