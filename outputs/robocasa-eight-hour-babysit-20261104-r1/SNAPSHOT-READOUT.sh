#!/usr/bin/env bash
set -Eeuo pipefail
: "${BABYSIT_CODE:?}" "${BABYSIT_COMMIT:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
case "$HOSTNAME" in
  *-master-0) INDEX=0; export BABYSIT_METHOD=ad ;;
  *-worker-0) INDEX=1; export BABYSIT_METHOD=dpt ;;
  *-worker-1) INDEX=2; export BABYSIT_METHOD=dit ;;
  *-worker-2) INDEX=3; export BABYSIT_METHOD=ic_iql ;;
  *) exit 2 ;;
esac
export BABYSIT_ATTEMPT=direct-snapshot-ID-qualification-01/$(printf 'node-%02d' "$INDEX")
export BABYSIT_SPLIT=ID BABYSIT_STAGE=qualification BABYSIT_COUNT=2
export BABYSIT_SNAPSHOT_MANIFEST=$ROOT/$RUN/inputs/direct-snapshot-01/$BABYSIT_METHOD.json
export BABYSIT_SECONDS_LIMIT=2550
START=$(date +%s)
bash "$BABYSIT_CODE/$RUN/READOUT.sh"
while (( $(date +%s) < START + BABYSIT_SECONDS_LIMIT )); do
  COUNT=$(find "$ROOT/$RUN/gpu/direct-snapshot-ID-qualification-01" -mindepth 2 -maxdepth 2 -name RESULT.json | wc -l)
  (( COUNT == 4 )) && exit 0
  sleep 10
done
exit 1
