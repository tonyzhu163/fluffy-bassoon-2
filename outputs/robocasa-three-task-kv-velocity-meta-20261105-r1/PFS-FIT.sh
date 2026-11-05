#!/usr/bin/env bash
set -Eeuo pipefail
export STEERING_INTERFACE=${1:?interface} STEERING_OBJECTIVE=${2:?objective}
case "$STEERING_INTERFACE" in kv_bridge|velocity_bridge) ;; *) exit 2 ;; esac
case "$STEERING_OBJECTIVE" in ad|dpt|dit|ic_iql) ;; *) exit 2 ;; esac
export STEERING_CODE
STEERING_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export STEERING_COMMIT
STEERING_COMMIT=$(git -C "$STEERING_CODE" rev-parse HEAD)
export STEERING_RESUME_CONTRACT_COMMIT=1c88834265cfadc511ba62cac48db0542ad0ebd1
export STEERING_MEMBER_CACHE=/mowenjie/fluffy-bassoon-2/inputs/steering-source-members-20261105-pfs01
export STEERING_MIN_FREE_GIB=220 STEERING_HOURS=4 STEERING_FIT_ONLY=1
export STEERING_ATTEMPT=${STEERING_ATTEMPT:-fit-pfs02-$STEERING_INTERFACE-$STEERING_OBJECTIVE}
export STEERING_QUALIFICATION=/mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/qualification-03/RESULT.json
bash "$STEERING_CODE/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/SINGLE.sh"
