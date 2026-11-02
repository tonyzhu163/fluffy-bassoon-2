"""Reconstruct the protocol-selected RGB collages for four retained causal prefixes."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

assert platform.system() == 'Linux'
assert len(sys.argv) == 2
code = Path(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path('/mowenjie/fluffy-bassoon-2')
run = 'robocasa-flora-drawer-sampled-frames-20261103-r1'
old = root/'outputs/robocasa-flora-drawer-grade-rubric-20261103-r1'
out = root/'outputs'/run
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
start = time.time()
source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
for name,digest in read(code/'outputs'/run/'EXPECTED-INPUTS.json').items():
    assert sha(old/name) == digest,name
assert read(old/'terminal-charge.json')['exit_code'] == 0
checks = read(old/'CPU-CHECKS.json')
assert checks['completed']
inputs = read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip() == inputs['official_code_commit']
sys.path[:0] = [str(code/'src'),inputs['official_code']]
import numpy as np
from PIL import Image,ImageDraw
from rewarder.batch_utils import core
from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
assert sha(code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py') == checks['unchanged_terminal_worker_sha256']
out.mkdir(exist_ok=False)
jobs = read(old/'PUBLIC-JOBS.json')
selected = [('000112',350),('000256',750),('000376',750),('000128',450)]
records = []
for case,control in selected:
    job = next(j for j in jobs if '-case-'+case+'-' in j['id'] and j['prefix_controls'] == control)
    assert all(sha(Path(v['path'])) == v['sha256'] for v in job['videos'])
    reader = _PrefixCollage(core.Collage([v['path'] for v in job['videos']]),control)
    try:
        assert reader.source.num_frames == 751 and reader.fps == 20.
        indices = core.terminal_frame_indices(reader.num_frames,reader.fps)
        assert len(indices) == checks['expected_sampled_frames'][job['id']] == 64
        assert min(indices) == 0 and max(indices) == control
        frames = [np.asarray(f) for f in reader.frames(indices)]
    finally: reader.release()
    assert all(f.dtype == np.uint8 and f.shape == (320,960,3) for f in frames)
    frame_records = []
    pages = []
    for offset in range(0,64,8):
        page = Image.new('RGB',(1920,1400),'white')
        draw = ImageDraw.Draw(page)
        for local in range(8):
            ordinal = offset+local
            frame = frames[ordinal]
            x,y = (local%2)*960,(local//2)*350
            page.paste(Image.fromarray(frame),(x,y+30))
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
            assert hashlib.sha256(np.asarray(page.crop((x,y+30,x+960,y+350))).tobytes()).hexdigest() == digest
            draw.text((x+5,y+8),f'Case {case}; sample {ordinal:02d}/63; control {indices[ordinal]}; {indices[ordinal]/20:.2f}s; left / wrist / right',fill='black')
            frame_records.append(dict(ordinal=ordinal,control=indices[ordinal],rgb_sha256=digest))
        path = out/f'case-{case}-prefix-{control}-page-{offset//8+1:02d}.png'
        page.save(path)
        pages.append(dict(file=path.name,sha256=sha(path),samples=list(range(offset,offset+8))))
    assert len(frame_records) == 64 and len(pages) == 8
    records.append(dict(case=case,id=job['id'],prefix_controls=control,source_videos=job['videos'],
        sampled_frames=frame_records,pages=pages,raw_frame_shape=[320,960,3],
        page_pixel_roundtrip=True,headers_for_human_only=True))
receipt = dict(completed=True,source_commit=source,cases=records,case_count=4,
    decoded_collages=256,pages=32,seconds=time.time()-start,cuda_visible_devices='',
    official_code_commit=inputs['official_code_commit'],unchanged_terminal_worker_sha256=checks['unchanged_terminal_worker_sha256'],
    model_calls=0,simulator_calls=0,policy_controls=0,motor_calls=0,optimizer_steps=0,
    limitation='Reconstructs protocol-selected RGB using frozen videos/reader/sampler. Previous GPU tensors were not saved, so no direct byte comparison to processed model tensors is claimed. Human pages never enter scoring.')
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
paths = sorted(out.iterdir())
manifest = dict(source_commit=source,files={p.name:sha(p) for p in paths})
archive = Path('/root/workspace/drawer-sampled-frames-r1.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths: z.write(p,p.name)
    z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(completed=True,cases=4,decoded_collages=256,pages=32,seconds=receipt['seconds'],
    archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),model_calls=0,simulator_calls=0)),flush=True)
