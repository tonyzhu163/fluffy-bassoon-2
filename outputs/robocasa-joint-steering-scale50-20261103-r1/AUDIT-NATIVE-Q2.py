"""Actual allocated-Linux CPU audit of retained native qualification artifacts."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-joint-steering-scale50-native-qualification-20261103-r2')
CODE=Path('/mowenjie/fluffy-bassoon-2/worktrees/joint-steering-scale50-gpu-20261103-r3')
PIN='8b9e1bbb3a73d55eda9bd9acdb30a85fee79d151'
assert socket.gethostname()=='aibox-rffe453e3884-5f96c78b88-rtvhb'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
OUTPUT=Path(sys.argv[1]);OUTPUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(CODE/'src'))
import numpy as np
import torch
import imageio_ffmpeg
from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import draw_seed
from fluffy_bassoon.meta_rl.joint_steering_ppo import TASKS

torch.set_num_threads(2)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()

def checked(reference):
 p=Path(reference['path']);assert p.is_file() and sha(p)==reference['sha256'];return p

def video(path):
 count,seconds=imageio_ffmpeg.count_frames_and_secs(str(path))
 assert count==601 and abs(seconds-601/20)<.11,(path,count,seconds)
 return dict(path=str(path),sha256=sha(path),frames=count,seconds=seconds)

started=time.time()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()==PIN
assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
calibration=json.loads((ROOT/'calibrate/CALIBRATION.json').read_text())
stress=json.loads((ROOT/'stress/RESULT.json').read_text())
assert calibration['completed'] and stress['completed'] and stress['status']=='passed'
assert stress['native_versions_unchanged'] and stress['child_release']['all_native_video_children_released']
assert not stress['escrow']['pending'] and stress['charged_controls']==57600
assert stress['bindings']['source_commit']==PIN==calibration['source_commit']
result=dict(completed=False,GPU_seconds=0,scientific_updates=0,source_commit=PIN,
 source_clean=True,physical_chunk_hashes=0,Gaussian_draws_and_summed_logps=0,
 arms={},started_unix=started,input_hashes={name:sha(ROOT/name) for name in ['calibrate/CALIBRATION.json','calibrate/RESULT.json','stress/RESULT.json','clock.json']})
all_videos=[]
for arm,width in [('velocity',32),('kv_all_layers',1152)]:
 data=stress['arms'][arm];q=calibration['arms'][arm]
 assert data['completed'] and data['lanes_closed_before_next_arm']
 assert data['execution']['peak_active_lanes']==48 and data['execution']['native_batch_sizes'][0]==48
 assert q['scale']['reference_bracketed'] and q['optimizer']['passed']
 selected=q['optimizer']['selected_optimizer_config']
 trial=next(t for t in q['optimizer']['trials'] if t['optimizer_config']==selected)
 assert trial['recovery']['next_update_exact'] and trial['recovery']['live_learner_unchanged']
 bank=torch.load(checked(data['history']),map_location='cpu',weights_only=False)
 assert len(bank['episodes'])==48 and len({(e['case']['task'],e['case']['native_seed']) for e in bank['episodes']})==48
 assert set(e['case']['task'] for e in bank['episodes'])==set(TASKS)
 for episode in bank['episodes']:
  assert episode['completed'] and episode['horizon']==600 and len(episode['chunks'])==12
  first=episode['first_success_step']
  expected_valid=12 if first is None else math.ceil(first/50)
  assert episode['valid_ppo_chunks']==expected_valid
  for index,reference in enumerate(episode['chunks']):
   path=checked(reference)
   with gzip.open(path,'rb') as f:r=pickle.load(f)
   assert r['status']=='completed' and r['executed_controls']==50 and r['chunk_index']==index
   assert r['source_commit']==PIN and r['case']==episode['case'] and r['policy_iteration']==0
   assert bool(r['valid_ppo'])==(index<expected_valid)
   assert r['reward']==float(first is not None and index==expected_valid-1)
   assert bool(r['done'])==bool(index==expected_valid-1)
   assert np.shape(r['executed_commands'])==(50,12) and np.isfinite(r['executed_commands']).all()
   assert np.shape(r['features'])==(2064,) and np.isfinite(r['features']).all()
   assert all(key in r for key in ['observation','raw_observation','next_observation','raw_next_observation','steering_mean','steering_std','native_noise'])
   seed=draw_seed(r['case'],index,'native_gaussian');assert seed==r['native_noise_seed']
   noise=torch.randn((50,32),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
   assert np.array_equal(noise.numpy(),r['native_noise'])
   mean,std,control=[torch.as_tensor(r[k],dtype=torch.float32) for k in ['steering_mean','steering_std','control']]
   assert mean.shape==std.shape==control.shape==(width,) and torch.isfinite(control).all() and (std>0).all()
   logp=float(torch.distributions.Normal(mean,std).log_prob(control).sum())
   assert abs(logp-r['log_prob'])<=max(.0005,abs(logp)*1e-6),(arm,logp,r['log_prob'])
   result['physical_chunk_hashes']+=1;result['Gaussian_draws_and_summed_logps']+=1
  for reference in [episode['initial'],episode['final'],episode['checkpoint']]:checked(reference)
  manifest=json.loads(Path(episode['video_manifest']).read_text())
  assert manifest['frame_clock']['frame_count']==601 and manifest['frame_clock']['resampled'] is False
  assert len(manifest['videos'])==3
  all_videos.extend(checked(ref) for ref in manifest['videos'])
 if arm=='kv_all_layers':
  targets=bank['qualification']['cache_targets'];assert len(targets)==36
  assert {(r['layer'],r['target']) for r in targets}=={(layer,target) for layer in range(18) for target in ['key','value']}
  assert all(t['original_cache_unchanged'] and t['differing_components']>0 and t['finite'] for t in targets)
 result['arms'][arm]=dict(episodes=48,policy_controls=28800,physical_chunks=576,
  native_batch_sizes=data['execution']['native_batch_sizes'],interface_spec=q['interface_spec'],optimizer_config=selected,
  fraction=q['scale']['selected_fraction'],action_RMS=q['scale']['selected_normalized_action_rms'],
  elapsed_seconds=data['elapsed_seconds'],controls_per_second=data['controls_per_second'],
  all36_KV_targets_verified=arm=='kv_all_layers',next_Adam_exact=True)
 del bank
with ThreadPoolExecutor(max_workers=2) as pool:videos=list(pool.map(video,all_videos))
assert len(videos)==288
(OUTPUT/'VIDEOS.json').write_text(json.dumps(videos,indent=2)+'\n')
result.update(completed=True,video_files=288,total_encoded_frames=sum(v['frames'] for v in videos),
 video_receipt_sha256=sha(OUTPUT/'VIDEOS.json'),finished_unix=time.time(),elapsed_seconds=time.time()-started)
(OUTPUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['completed','physical_chunk_hashes','Gaussian_draws_and_summed_logps','video_files','total_encoded_frames','GPU_seconds','elapsed_seconds']}))
