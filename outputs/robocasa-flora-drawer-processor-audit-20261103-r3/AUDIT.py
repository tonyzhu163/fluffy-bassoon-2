"""Reconstruct official CPU video features for four frozen OpenDrawer prefixes."""
import faulthandler
import gc
import site
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

assert platform.system() == 'Linux' and len(sys.argv) == 2
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('PYTHONNOUSERSITE') == '1' and site.ENABLE_USER_SITE is False
faulthandler.enable()
faulthandler.dump_traceback_later(60,repeat=True)
print('AUDIT_START',time.time(),flush=True)
code = Path(sys.argv[1])
root = Path('/mowenjie/fluffy-bassoon-2')
run = 'robocasa-flora-drawer-processor-audit-20261103-r3'
old = root/'outputs/robocasa-flora-drawer-target-hints-20261103-r1'
visual = root/'outputs/robocasa-flora-drawer-sampled-frames-20261103-r1'
out = root/'outputs'/run
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p,d: p.write_text(json.dumps(d,indent=2)+'\n')
start = time.time()
source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
for path,digest in read(code/'outputs'/run/'EXPECTED-INPUTS.json').items():
    assert sha(Path(path)) == digest,path
survey = read(code/'outputs'/run/'SOURCE-SURVEY.json')
for record in survey.values():
    assert sha(Path(record['path'])) == record['sha256'],record['path']
assert read(old/'terminal-charge.json')['exit_code'] == 0
checks = read(old/'CPU-CHECKS.json')
assert checks['completed'] and checks['new_outputs'] == 82
inputs = read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
assert sha(Path(inputs['model_receipt']['path'])) == inputs['model_receipt']['sha256']
assert read(Path(inputs['model_receipt']['path']))['completed']
assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip() == inputs['official_code_commit']
assert sha(code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py') == checks['unchanged_terminal_worker_sha256']
sys.path[:0] = [str(code/'src'),inputs['official_code']]
out.mkdir(exist_ok=False)
print('FROZEN_INPUTS_VERIFIED',time.time(),flush=True)
import numpy as np
import torch
print('TORCH_IMPORTED',time.time(),flush=True)
from PIL import Image,ImageDraw
from transformers.video_utils import VideoMetadata
print('TRANSFORMERS_IMPORTED',time.time(),flush=True)
from rewarder.batch_utils import core,engine
from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage

torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
processor = engine.load_processor(inputs['model'])
print('PROCESSOR_READY',time.time(),flush=True)
vp = processor.video_processor
assert type(processor).__name__ == 'Qwen3VLProcessor'
assert type(vp).__name__ == 'Qwen3VLVideoProcessor'
assert vp.patch_size == 16 and vp.temporal_patch_size == vp.merge_size == 2
channel_mean = np.broadcast_to(np.asarray(vp.image_mean,dtype=np.float64),(3,)).tolist()
channel_std = np.broadcast_to(np.asarray(vp.image_std,dtype=np.float64),(3,)).tolist()
print('NORMALIZATION',json.dumps(dict(mean=channel_mean,std=channel_std,mean_type=type(vp.image_mean).__name__,std_type=type(vp.image_std).__name__)),flush=True)
assert engine.COLLAGE_VIDEO_SIZE == dict(shortest_edge=4096,longest_edge=29360128)
write(out/'PROCESSOR.json',dict(processor_class=type(processor).__name__,video_processor=vp.to_dict(),
    normalization=dict(mean=channel_mean,std=channel_std,mean_type=type(vp.image_mean).__name__,std_type=type(vp.image_std).__name__,matches_saved_model_config=channel_mean==channel_std==[0.5,0.5,0.5]),
    request_kwargs=dict(do_sample_frames=False,size=engine.COLLAGE_VIDEO_SIZE),
    versions={p:importlib.metadata.version(p) for p in ('transformers','torch','torchvision','vllm','numpy')},
    source_survey=survey,official_code_commit=inputs['official_code_commit'],model_revision=inputs['model_revision']))
jobs = read(old/'PUBLIC-JOBS.json')
frame_reference = read(old/'FRAME-REFERENCE.json')
prepared = {p['id']:p['messages'] for p in read(old/'PREPARED-PROMPTS.json')}
assert sha(old/'PREPARED-PROMPTS.json') == checks['prompts_sha256']
actual_prompts = {p['id']:p for file in (old/'scoring').glob('*.prompts.json') for p in read(file)}
audited = {c['id']:c for c in read(visual/'RECEIPT.json')['cases']}
selected = [('000112',350),('000256',750),('000376',750),('000128',450)]
records = []

def unpatch(features,grid):
    gt,gh,gw = grid
    p,t,m = vp.patch_size,vp.temporal_patch_size,vp.merge_size
    assert tuple(features.shape) == (gt*gh*gw,3*t*p*p)
    packed = features.float().reshape(gt,gh//m,gw//m,m,m,3,t,p,p)
    normalized = packed.permute(0,6,5,1,3,7,2,4,8).reshape(gt*t,3,gh*p,gw*p)
    mean = torch.tensor(channel_mean).reshape(1,3,1,1)
    std = torch.tensor(channel_std).reshape(1,3,1,1)
    rgb_float = ((normalized*std+mean)/vp.rescale_factor).permute(0,2,3,1).contiguous()
    return rgb_float.round().clamp(0,255).to(torch.uint8).numpy(),rgb_float

for case,control in selected:
    job = next(j for j in jobs if '-case-'+case+'-' in j['id'] and j['prefix_controls'] == control)
    uid = job['id'];label_id = uid+'/native_boundary'
    assert all(sha(Path(v['path'])) == v['sha256'] for v in job['videos'])
    reader = _PrefixCollage(core.Collage([v['path'] for v in job['videos']]),control)
    try:
        indices = core.terminal_frame_indices(reader.num_frames,reader.fps)
        frames = np.stack(reader.frames(indices))
        metadata = core.video_metadata(indices,reader.fps,reader.num_frames)
    finally:reader.release()
    hashes = [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
    assert frames.shape == (64,320,960,3) and frames.dtype == np.uint8
    assert list(indices) == frame_reference[uid]['indices']
    assert hashes == frame_reference[uid]['rgb_sha256']
    assert hashes == [f['rgb_sha256'] for f in audited[uid]['sampled_frames']]
    messages = prepared[label_id]
    prompt = engine.render(processor,messages,thinking=False)
    assert hashlib.sha256(prompt.encode()).hexdigest() == actual_prompts[label_id]['rendered_sha256']
    assert hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest() == actual_prompts[label_id]['messages_sha256']
    native_metadata = {k:v for k,v in metadata.items() if k != 'do_sample_frames'}
    kwargs = dict(do_sample_frames=False,size=engine.COLLAGE_VIDEO_SIZE,return_tensors='pt')
    # Matches the nested pre-sampled video/metadata inputs in the pinned vLLM
    # Qwen3VLMultiModalProcessor._call_hf_processor implementation.
    full = processor(text=[prompt],videos=[[frames]],
        video_metadata=[[VideoMetadata(**native_metadata)]],**kwargs)
    features,grid = full['pixel_values_videos'],full['video_grid_thw'][0].tolist()
    assert features.device.type == 'cpu' and features.dtype == torch.float32
    direct = vp(videos=[[frames]],video_metadata=[[VideoMetadata(**native_metadata)]],
        return_metadata=True,**kwargs)
    assert torch.equal(features,direct['pixel_values_videos'])
    assert torch.equal(full['video_grid_thw'],direct['video_grid_thw'])
    observed_meta = direct['video_metadata'][0]
    assert list(observed_meta.frames_indices) == list(indices) and observed_meta.fps == 20.
    assert grid == [32,20,60]
    restored,rgb_float = unpatch(features,grid)
    restored_hashes = [hashlib.sha256(f.tobytes()).hexdigest() for f in restored]
    assert restored_hashes == hashes
    max_float_error = float((rgb_float-torch.from_numpy(frames).float()).abs().max())
    del rgb_float
    bf16 = features.to(torch.bfloat16)
    rounded_bf16,bf16_float = unpatch(bf16,grid)
    bf16_max_float_error = float((bf16_float-torch.from_numpy(frames).float()).abs().max())
    bf16_differing_channels = int(np.count_nonzero(rounded_bf16 != frames))
    bf16_max_byte_error = int(np.abs(rounded_bf16.astype(np.int16)-frames.astype(np.int16)).max())
    image = Image.new('RGB',(960,700),'white');draw = ImageDraw.Draw(image)
    draw.text((5,5),f'Case {case}; control {control}; original RGB (top), CPU processor round-trip (bottom)',fill='black')
    image.paste(Image.fromarray(frames[-1]),(0,30))
    draw.text((5,355),'All 64 reconstructed frames match source RGB byte for byte; labels are for this audit only.',fill='black')
    image.paste(Image.fromarray(restored[-1]),(0,380))
    path = out/f'case-{case}-prefix-{control}-roundtrip.png';image.save(path)
    record = dict(case=case,id=uid,prefix_controls=control,indices=list(indices),metadata=metadata,
        raw_shape=list(frames.shape),video_grid_thw=grid,feature_shape=list(features.shape),
        merged_vision_tokens=int(np.prod(grid)//vp.merge_size**2),feature_dtype=str(features.dtype),
        feature_sha256=hashlib.sha256(features.numpy().tobytes()).hexdigest(),
        bf16_feature_sha256=hashlib.sha256(bf16.view(torch.uint8).numpy().tobytes()).hexdigest(),
        full_and_direct_video_processor_equal=True,raw_rgb_sha256=hashes,roundtrip_rgb_sha256=restored_hashes,
        raw_byte_differences=0,float_roundtrip_max_rgb_error=max_float_error,
        bf16_reconstruction_differing_channels=bf16_differing_channels,bf16_reconstruction_max_byte_error=bf16_max_byte_error,
        bf16_reconstruction_max_rgb_float_error=bf16_max_float_error,view_width=320,view_height=320,
        actual_previous_rendered_prompt_matches=True,rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        image_file=path.name,image_sha256=sha(path))
    records.append(record)
    print(json.dumps(dict(case=case,control=control,grid=grid,shape=list(features.shape),rgb_byte_differences=0,
        bf16_differing_channels=bf16_differing_channels)),flush=True)
    del frames,restored,rounded_bf16,bf16_float,full,direct,features,bf16,image
    gc.collect()
assert not torch.cuda.is_initialized()
faulthandler.cancel_dump_traceback_later()
receipt = dict(completed=True,source_commit=source,official_code_commit=inputs['official_code_commit'],
    model_revision=inputs['model_revision'],cases=records,case_count=4,reconstructed_frames=256,
    raw_rgb_matches_previous_scoring_and_visual_audit=True,all_float32_roundtrips_byte_exact=True,
    seconds=time.time()-start,cuda_visible_devices='',torch_cuda_initialized=False,
    model_weight_loads=0,model_inference_calls=0,simulator_calls=0,policy_controls=0,motor_calls=0,optimizer_steps=0,
    limitation='CPU reconstruction of the pinned HF processor called by the pinned vLLM implementation, with the same raw inputs and options. Previous GPU processor/encoder tensors were not saved. This does not verify semantic retention after learned visual embeddings, language-model grounding, or independently qualify a reward.')
write(out/'RECEIPT.json',receipt)
files = sorted(p for p in out.iterdir() if p.is_file())
manifest = dict(source_commit=source,files={p.name:sha(p) for p in files})
archive = Path('/root/workspace/drawer-processor-audit-r3.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.name)
    z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(completed=True,cases=4,reconstructed_frames=256,seconds=receipt['seconds'],
    archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive))),flush=True)
