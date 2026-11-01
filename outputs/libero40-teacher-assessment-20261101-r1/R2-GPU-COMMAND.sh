bash <<'TEACHER_ASSESSMENT_GPU_R2'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=9d29a805cae2c129ffc088af4eed81b3a38d2769
TASK_LEGACY_SOURCE=7ef8a62058773201e85c8b2ec26bf3847ff15026
TASK_RUN=libero40-teacher-assessment-20261101-r1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_CODE=$TASK_ROOT/worktrees/libero40-teacher-assessment-repair-20261101-r1
TASK_OUT=$TASK_ROOT/outputs/libero40-teacher-assessment-20261101-r2
TASK_CLOCKROOT=$TASK_ROOT/outputs/$TASK_RUN/clock
TASK_CPU=$TASK_ROOT/outputs/libero40-teacher-assessment-repair-cpu-20261101-r1/cpu-r1
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
export TASK_LEGACY_SOURCE TASK_CLOCKROOT TASK_CPU TASK_STARTED TASK_SOURCE TASK_RUN TASK_ROOT TASK_OLD TASK_CODE TASK_OUT TASK_MODEL TASK_PROBE TASK_OPENPI TASK_DSRL TASK_LIBERO
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test -f "$TASK_CPU/summary.json"
test -f "$TASK_CLOCKROOT/worker-time.json"
test ! -e "$TASK_OUT/data"
test ! -e "$TASK_OUT/exit-code.txt"
exec 9> "$TASK_OUT/assessment-launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$TASK_CODE/src
TASK_ATTEMPT=$("$TASK_MODEL" -B - <<'PY'
import json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_teacher_assessment import worker_contract
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
root=Path(os.environ['TASK_OUT']);cpu=json.loads((Path(os.environ['TASK_CPU'])/'summary.json').read_text())
assert cpu['passed'] is True and cpu['source_commit']==os.environ['TASK_SOURCE'] and cpu['JUnit']['tests']=='30'
print(begin(Path(os.environ['TASK_CLOCKROOT']),worker_contract(os.environ['TASK_LEGACY_SOURCE'],os.environ['TASK_RUN']),slice_seconds=9360,worker_seconds=10800,resume=True,now=int(os.environ['TASK_STARTED'])))
PY
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
task_cleanup() {
 TASK_RC=$?
 trap - EXIT
 PYTHONPATH=$TASK_CODE/src "$TASK_MODEL" -B - "$TASK_CLOCKROOT" "$TASK_ATTEMPT" "$TASK_RC" <<'PY' || true
import json,sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
print(json.dumps(finish(sys.argv[1],sys.argv[2],int(sys.argv[3]))))
PY
 printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training --baige-container --baige-exclusive-training --output "$TASK_OUT/admission/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/admission/gpu-probe.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$TASK_PROBE" -B - <<'PY'
import json,os,socket
from pathlib import Path
r=Path(os.environ['TASK_OUT'])/'admission'
paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective']
raw={p:Path(p).read_text().strip() for p in paths};q,p=raw['/sys/fs/cgroup/cpu.max'].split();cores=int(q)/int(p);mem=int(raw['/sys/fs/cgroup/memory.max']);shm=os.statvfs('/dev/shm');shared=shm.f_blocks*shm.f_frsize
assert socket.gethostname().startswith('job-') and cores==16 and mem==96*1024**3 and shared==8*1024**3
(r/'actual-resources.json').write_text(json.dumps(dict(hostname=socket.gethostname(),CPU=cores,memory_bytes=mem,shared_memory_bytes=shared,raw_cgroup=raw,scheduler_visibility={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']},no_added_GPU_namespace=True),indent=2)+'\n')
PY
export PYTHONPATH=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1:$TASK_CODE/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_TICKET_SHA=$("$TASK_MODEL" -B - <<'PY'
import hashlib,json,os,shutil,socket,time
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_teacher_assessment import FORMAT,RUNTIME_FILES,worker_contract
root,code,attempt=map(Path,(os.environ['TASK_OUT'],os.environ['TASK_CODE'],os.environ['TASK_ATTEMPT']))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
assets=code/'outputs/libero40-teacher-assessment-20261101-r1'
refs={k:ref(assets/v) for k,v in {'grant':'GRANT.json','authority':'AUTHORITY.json','protocol':'PROTOCOL.json','asset_binding':'ASSET-BINDING.json','actor_table':'ACTOR-TABLE.json','inventory':'METADATA-INVENTORY.json'}.items()}
free=shutil.disk_usage(root).free;required=500_000_000_000+(128+64+32)*1024**3
assert free>=required
p=root/'write-probe-gpu.txt'
with p.open('x') as f:f.write('assessment-owned-write-probe\n');f.flush();os.fsync(f.fileno())
resources=json.loads((root/'admission/actual-resources.json').read_text());probe=json.loads((root/'admission/gpu-probe.json').read_text());assert probe['status']=='passed' and len(probe['devices'])==1
allocation=dict(GPU=1,CPU=16,memory_GiB=96,shared_memory_GiB=8,automatic_retry=False,exclusive=True)
ticket=dict(format=FORMAT,run_id=os.environ['TASK_RUN'],admitted=True,source_commit=os.environ['TASK_SOURCE'],hostname=socket.gethostname(),job_id=socket.gethostname().removesuffix('-master-0'),clock_root=os.environ['TASK_CLOCKROOT'],clock_contract=worker_contract(os.environ['TASK_LEGACY_SOURCE'],os.environ['TASK_RUN']),resource_attempt=str(attempt),output=str(root/'data'),driver_deadline_unix=int(os.environ['TASK_STARTED'])+9360-120,gpu_uuid=probe['devices'][0]['uuid'],gpu_probe=ref(root/'admission/gpu-probe.json'),allocation=allocation,refs=refs,runtime_files={n:sha(code/'src/fluffy_bassoon/meta_rl'/n) for n in RUNTIME_FILES},actual_resources=ref(root/'admission/actual-resources.json'),CPU_qualification=ref(Path(os.environ['TASK_CPU'])/'summary.json'),storage=dict(free_bytes=free,minimum_free_with_reserves=required,output_cap_bytes=32<<30),reviewed_latest_main='a7107b5a9d9ba23803669b2367ec56f01da32516',carried_qualification_no_extra_episode=True)
proof=root/'admission/PRIOR-STORAGE-PROOF.json'
assert sha(proof)=='27c933991329afb98f69b7914e17d81a2e0b77d5f934c46a1a4781a1c8310dc4'
clockroot=Path(os.environ['TASK_CLOCKROOT'])
assert (clockroot/'worker-time.json').stat().st_size<32768 and (attempt/'attempt.json').stat().st_size<8192
packet=code/'outputs/libero40-teacher-assessment-repair-code-20261101-r1'
ticket.update(repair_profile=ref(packet/'PROFILE.json'),repair_decision=ref(code/'outputs/libero40-teacher-assessment-20261101-r1/REPAIR-DECISION-R2.json'),repair_CPU_qualification=ref(Path(os.environ['TASK_CPU'])/'summary.json'),prior_storage_proof=ref(proof),original_source=os.environ['TASK_LEGACY_SOURCE'],remaining_unstarted_roles=158,aborted161_never_replayed=True)
p=root/'admission/ticket.json';p.write_text(json.dumps(ticket,indent=2)+'\n');print(sha(p))
PY
)
TASK_TIMEOUT=$((TASK_STARTED+9360-$(date +%s)-60))
test "$TASK_TIMEOUT" -gt 300
set +e
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" -B -m fluffy_bassoon.meta_rl.vla_source40_teacher_assessment --output "$TASK_OUT/data" --ticket "$TASK_OUT/admission/ticket.json" --ticket-sha256 "$TASK_TICKET_SHA" --expected-source "$TASK_SOURCE" --gpu-probe "$TASK_OUT/admission/gpu-probe.json" --actor-table "$TASK_CODE/outputs/libero40-teacher-assessment-20261101-r1/ACTOR-TABLE.json" --metadata-inventory "$TASK_CODE/outputs/libero40-teacher-assessment-20261101-r1/METADATA-INVENTORY.json" --openpi-root "$TASK_OPENPI" --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --dsrl-root "$TASK_DSRL" --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python --simulator-pythonpath "$TASK_CODE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" --port 8783
TASK_RC=$?
exit "$TASK_RC"
TEACHER_ASSESSMENT_GPU_R2
