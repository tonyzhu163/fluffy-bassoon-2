#!/usr/bin/env bash
set -Eeuo pipefail
: "${BABYSIT_CODE:?}" "${BABYSIT_COMMIT:?}" "${BABYSIT_METHOD:?}" "${BABYSIT_SPLIT:?}" "${BABYSIT_GROUP:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
case "$HOSTNAME" in
  *-master-0) INDEX=0 ;;
  *-worker-0) INDEX=1 ;;
  *-worker-1) INDEX=2 ;;
  *-worker-2) INDEX=3 ;;
  *) exit 2 ;;
esac
case "$BABYSIT_METHOD" in
  native_initial|native_final) WORKERS=2; export BABYSIT_MEMORY_MODE=full ;;
  ad|dpt|dit|ic_iql)
    WORKERS=4
    if (( INDEX < 2 )); then export BABYSIT_MEMORY_MODE=full; else export BABYSIT_MEMORY_MODE=episode; fi ;;
  *) exit 2 ;;
esac
(( INDEX < WORKERS ))
export BABYSIT_ATTEMPT=$BABYSIT_GROUP/$(printf 'node-%02d' "$INDEX")
export BABYSIT_SHARDS=2 BABYSIT_SHARD_INDEX=$((INDEX%2))
export BABYSIT_STAGE=final BABYSIT_COUNT=50
export BABYSIT_SECONDS_LIMIT=${BABYSIT_SECONDS_LIMIT:-4200}
START=$(date +%s)
bash "$BABYSIT_CODE/$RUN/READOUT.sh"
# Retain every replica until all complete-shard receipts exist. A successful
# master must not tear down another replica's remaining real episodes.
while (( $(date +%s) < START + BABYSIT_SECONDS_LIMIT )); do
  COUNT=$(find "$ROOT/$RUN/gpu/$BABYSIT_GROUP" -mindepth 2 -maxdepth 2 -name RESULT.json | wc -l)
  (( COUNT == WORKERS )) && exit 0
  sleep 10
done
exit 1
