# ONLY inside the same new120-worker-second software allocation, after its
# actual0GPU/cgroup/Git/import proof and focused tests. Do not renew its clock.
"${REPAIR_CPU_PYTHON:?qualified source Python3.11}" -B -m fluffy_bassoon.meta_rl.vla_source40_teacher_repair \
  --progress /mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1/data/assessment-recovery.pkl \
  --table outputs/libero40-teacher-assessment-20261101-r1/ACTOR-TABLE.json \
  --old-ticket outputs/libero40-teacher-assessment-repair-code-20261101-r1/ORIGINAL-TICKET.json \
  --closed-clock outputs/libero40-teacher-assessment-repair-code-20261101-r1/ORIGINAL-CLOSED-CLOCK.json \
  --output "${REPAIR_TINY_DRYRUN_JSON:?fresh small output JSON}"
