# Existing owner only: inside the one admitted exclusive1GPU allocation, after
# the existing begin clock/probe and actual ticket. This is NOT a submit script.
"${ASSESSMENT_GPU_PYTHON:?qualified OpenPI Python3.11}" -B -m fluffy_bassoon.meta_rl.vla_source40_teacher_assessment \
  --output "${ASSESSMENT_DATA_OUTPUT:?new assessment phase/data}" \
  --ticket "${ASSESSMENT_TICKET:?actual owner-issued immutable JSON}" \
  --ticket-sha256 "${ASSESSMENT_TICKET_SHA:?actual JSON byte SHA}" \
  --expected-source "${ASSESSMENT_SOURCE:?reviewed immutable execution pin}" \
  --gpu-probe "${ASSESSMENT_GPU_PROBE:?actual own one-GPU proof}" \
  --actor-table outputs/libero40-teacher-assessment-20261101-r1/ACTOR-TABLE.json \
  --metadata-inventory outputs/libero40-teacher-assessment-20261101-r1/METADATA-INVENTORY.json \
  --openpi-root "${ASSESSMENT_OPENPI_ROOT:?actual qualified215abf checkout}" \
  --checkpoint /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1 \
  --dsrl-root "${ASSESSMENT_DSRL_ROOT:?actual qualified7f4893 checkout}" \
  --simulator-python "${ASSESSMENT_SIMULATOR_PYTHON:?qualified LIBERO Python}" \
  --simulator-pythonpath "${ASSESSMENT_SIMULATOR_PYTHONPATH:?exact source and original LIBERO paths}" \
  --port "${ASSESSMENT_LOOPBACK_PORT:-8783}"
