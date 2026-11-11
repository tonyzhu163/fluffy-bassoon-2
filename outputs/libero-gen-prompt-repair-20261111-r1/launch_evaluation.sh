#!/usr/bin/env bash
# Eight paired prompt episodes: one GPU, <=60 minutes; frozen BPP weights.
set -euo pipefail
TASK_PROJECT=${1:?shared project root}
TASK_CODE=${2:?immutable project checkout}
TASK_COMMIT=${3:?expected full project SHA}
TASK_ID=${4:?fresh qualification identity such as qual01}
TASK_TEACHER_ID=${5:?completed teacher output identity}
TASK_ALIGNMENT=${6:?pre or post from native source alignment}
TASK_SECOND_TEACHER_ID=${7:-$TASK_TEACHER_ID}
[[ $TASK_ID =~ ^[a-z0-9-]+$ && $HOSTNAME == job-*-master-0 ]]
[[ $TASK_TEACHER_ID =~ ^[a-z0-9-]+$ && ( $TASK_ALIGNMENT == pre || $TASK_ALIGNMENT == post ) ]]
[[ $TASK_SECOND_TEACHER_ID =~ ^[a-z0-9-]+$ ]]
TASK_FILES="$TASK_CODE/outputs/libero-gen-prompt-repair-20261111-r1"
TASK_PARENT="$TASK_PROJECT/outputs/libero-gen-prompt-repair-20261111-r1"
mkdir -p "$TASK_PARENT"
TASK_RUN="$TASK_PARENT/$TASK_ID"
TASK_SETUP="$TASK_PROJECT/setups/libero-gen-history-pilot-20261111-r1"
TASK_PY="$TASK_SETUP/env/bin/python"
mkdir "$TASK_RUN"
TASK_STARTED=$(date +%s)
trap 'TASK_RC=$?; TASK_ENDED=$(date +%s); printf "{\"exit_code\":%s,\"started\":%s,\"ended\":%s,\"gpus\":1,\"hostname\":\"%s\"}\n" "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$HOSTNAME" > "$TASK_RUN/clock.json"' EXIT
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_COMMIT"
git -C "$TASK_CODE" diff --quiet HEAD
export FLUFFY_GPU_EXECUTION=baige
export PYTHONPATH="$TASK_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
# The private GLVND dispatch packages repair the CPU image's missing libGL/EGL.
# NVIDIA vendor libraries remain supplied by the scheduler container runtime.
TASK_GL_LIB="$TASK_SETUP/os-libs01/prefix/usr/lib/x86_64-linux-gnu"
TASK_TORCH_LIB="$TASK_SETUP/env/lib/python3.10/site-packages/torch/lib"
test -f "$TASK_GL_LIB/libEGL.so.1"
test -f "$TASK_GL_LIB/libGL.so.1"
export LD_LIBRARY_PATH="$TASK_GL_LIB:$TASK_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$TASK_SETUP/huggingface-cache"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export MPLCONFIGDIR="$TASK_RUN/matplotlib"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Preserve the scheduler's CUDA_VISIBLE_DEVICES and container isolation.
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  TASK_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $TASK_UUID == GPU-* && $TASK_UUID != *$'\n'* && $TASK_UUID == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$TASK_PY" "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$TASK_RUN/gpu-probe.json"
"$TASK_PY" - "$TASK_FILES" "$TASK_SETUP" "$TASK_PROJECT" "$TASK_RUN" "$TASK_TEACHER_ID" "$TASK_ALIGNMENT" "$TASK_SECOND_TEACHER_ID" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys,time
files,setup,project,out=map(Path,sys.argv[1:5])
teachers=[project/'outputs/libero-gen-prompt-repair-20261111-r1'/identity/'teacher' for identity in [sys.argv[5],sys.argv[7]]]
alignment=sys.argv[6]
started=time.time()
split='libero_spatial_selected_combinations_inverse_view'
tasks=['pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate',
       'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2']
geometry=project/'outputs/libero-gen-custom-layouts-20261111-r1/geometry02/geometry'
admission=files.parent/'libero-gen-custom-layouts-20261111-r1/admission.json'
assert all(json.loads((teacher/'result.json').read_text())['complete'] for teacher in teachers)

def run(command, log_name):
 remaining=3300-(time.time()-started)
 assert remaining>0, 'Bounded evaluation allocation exhausted'
 with (out/log_name).open('w') as log:
  subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=True)

code=124
try:
 for condition in ['original','repaired']:
  assets=out/(condition+'-assets')
  assets.mkdir()
  (assets/'model').symlink_to(setup/'assets/model', target_is_directory=True)
  for candidate,task in enumerate(tasks):
   source=setup/'assets/dataset/demonstration_data'/split/(task+'_demo.hdf5')
   target=assets/'dataset/demonstration_data'/split/(task+'_demo.hdf5')
   command=[sys.executable,'-Bu',str(files/'pack_prompt.py'),'--source-hdf5',str(source),'--output',str(target)]
   if condition=='original':
    command+=['--copy-reference']
   else:
    folder=teachers[candidate]/f'layout00-{candidate}'
    command+=['--trajectory',str(folder/'trajectory.npz'),'--model-xml',str(folder/'model.xml'),
              '--alignment',alignment,'--selected-bowl',str(candidate),'--stop-at-first-native-success']
   run(command,f'pack-{condition}-{candidate}.log')
 # Prepare both prompt sets before spending time on the policy comparison.
 for condition in ['original','repaired']:
  command=[sys.executable,'-Bu',str(files/'evaluate.py'),
           '--source',str(project/'cache/bpp-58125e6-20261111-r1'),
           '--assets',str(out/(condition+'-assets')),'--output',str(out/condition),
           '--geometry',str(geometry),'--admission',str(admission),
           '--admission-sha256',hashlib.sha256(admission.read_bytes()).hexdigest(),
           '--indices','0,1','--condition',condition]
  run(command,condition+'.log')
 results=[json.loads((out/condition/'result.json').read_text()) for condition in ['original','repaired']]
 assert all(r['complete'] and r['indices']==[0,1] for r in results)
 assert results[0]['checkpoint_sha256']==results[1]['checkpoint_sha256']
 rows=[]
 for candidate in range(2):
  old,new=[r['results'][candidate] for r in results]
  assert old['policy_seeds']==new['policy_seeds']
  for position,index in enumerate([0,1]):
   a,b=old['records'][position],new['records'][position]
   for key in ['index','candidate','initial_state_sha256','initial_observation_hashes','selected_physical_index']:
    assert a[key]==b[key], f'Unpaired {key}'
   rows.append({'layout':index,'candidate':candidate,'seed':old['policy_seeds'][position],
                'original':{k:a[k] for k in ['selected_skill_success','native_any','actual_controls']},
                'repaired':{k:b[k] for k in ['selected_skill_success','native_any','actual_controls']}})
 (out/'comparison.json').write_text(json.dumps({'complete':True,'teachers':[str(p) for p in teachers],
    'alignment':alignment,'checkpoint_sha256':results[0]['checkpoint_sha256'],
    'paired_states_observations_and_policy_seeds':True,'weights_changed':False,
    'rows':rows},indent=2)+'\n')
 code=0
finally:
 (out/'process-clock.json').write_text(json.dumps({'started':started,'ended':time.time(),'exit_code':code,'gpu_count':1,'process_gpu_hours':(time.time()-started)/3600},indent=2)+'\n')
raise SystemExit(code)
PY
