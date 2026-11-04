#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-flow-sde-critic-parity-20261104-r1
TASK_CONTRACT=$TASK_RUN/comparison-input-r1/DEPLOYED.json
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_OUT=$TASK_RUN/comparison-r1
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
TASK_STARTED=$(date +%s)
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_FINISHED=$(date +%s)
  printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":8,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
    "$TASK_STARTED" "$TASK_FINISHED" "$((8*(TASK_FINISHED-TASK_STARTED)))" "$TASK_RC" > "$TASK_OUT/clock.json"
  exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
TASK_PREPARED=$TASK_RUN/comparison-input-r1/PREPARED.json
"$TASK_PY" -B - "$TASK_PREPARED" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));assert d['completed'] and len(d['files'])==22
PY
export TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
export ROBOCASA_FLOW_TASK=PickPlaceDrawerToCounter
export ROBOCASA_FLOW_TRAIN_SEED=2300000 ROBOCASA_FLOW_EVAL_SEED=630100
export ROBOCASA_FLOW_ACTOR_BATCH_DROP='["forward_inputs.observation/image","forward_inputs.observation/wrist_image","forward_inputs.observation/extra_view_image","forward_inputs.chains"]'
export ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true
export ROBOCASA_SIM_PPO_RESUME=$TASK_RUN/comparison-input-r1/global_step_0
export ROBOCASA_SIM_PPO_BEFORE_REFERENCE=$TASK_RUN/comparison-input-r1/BEFORE-REFERENCE.json
export ROBOCASA_SIM_PPO_BEFORE_SHA256=$("$TASK_PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["before_reference_sha256"])' "$TASK_PREPARED")
export TASK_GPUS=8 TASK_CPUS=168 TASK_RAM_GIB=1280 TASK_SHM_GIB=32 TASK_SEGMENT=comparison
for CRITIC_COMPARISON_ARM in baseline fixed; do
  export CRITIC_COMPARISON_ARM
  TASK_NOW=$(date +%s)
  TASK_REMAINING=$((2400-(TASK_NOW-TASK_STARTED)-60))
  if (( TASK_REMAINING < 900 )); then echo 'Insufficient declared time for another arm'; exit 124; fi
  export TASK_WALL_SECONDS=$((TASK_REMAINING<1140 ? TASK_REMAINING : 1140))
  export TASK_ATTEMPT=critic-comparison-$CRITIC_COMPARISON_ARM-r1
  export TASK_OUTPUT=$TASK_OUT/$CRITIC_COMPARISON_ARM
  export ROBOCASA_SIM_PPO_ACCOUNTING=$TASK_OUT/accounting-$CRITIC_COMPARISON_ARM
  export TASK_VENDOR=$("$TASK_PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"])' "$TASK_CONTRACT" "$CRITIC_COMPARISON_ARM")
  export TASK_VENDOR_PIN=$("$TASK_PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]["commit"])' "$TASK_CONTRACT" "$CRITIC_COMPARISON_ARM")
  bash "$TASK_META/outputs/robocasa-flow-sde-critic-parity-20261104-r1/COMPARE-ARM.sh"
  "$TASK_PY" -B - "$TASK_OUTPUT" <<'PY'
import json,sys,pathlib
r=pathlib.Path(sys.argv[1]);p=json.loads((r/'progress.json').read_text())
assert p['completed_iterations']==3
assert json.loads((r/'after-evaluation.json').read_text())['completed']
assert json.loads((r/'clock.json').read_text())['exit_code']==0
PY
  # Each job container is exclusive. Wait for its worker processes to exit;
  # never send signals by broad process name between the two arms.
  "$TASK_PY" -B - "$TASK_OUTPUT" <<'PY'
import os,pathlib,sys,time
root=sys.argv[1]
def active():
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            env=(p/'environ').read_bytes().split(b'\0')
            if ('ROBOCASA_SIM_PPO_OUTPUT='+root).encode() in env and ('python' in cmd) and 'COMPARE-COMMAND' not in cmd:
                found.append(dict(pid=int(p.name),command=cmd[:500]))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found
deadline=time.monotonic()+30
while active() and time.monotonic()<deadline:time.sleep(1)
left=active();assert not left, ('Prior arm workers still active; preserve and stop comparison',left)
PY
done
