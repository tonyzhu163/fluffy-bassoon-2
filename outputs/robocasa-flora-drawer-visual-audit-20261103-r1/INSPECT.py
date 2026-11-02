"""Retained-video contact sheets for human inspection only; no model or simulator."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

assert platform.system() == 'Linux'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path('/mowenjie/fluffy-bassoon-2')
run = 'robocasa-flora-drawer-visual-audit-20261103-r1'
out = root/'outputs'/run
assert len(sys.argv) == 2
source = sys.argv[1]
start = time.time()
out.mkdir(exist_ok=False)
sys.path.insert(0, str(root/'deps/nvidia-flora-129a2ada-20261101-r1/repo'))
import numpy as np
from PIL import Image, ImageDraw
from rewarder.batch_utils import core

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
public = root/'outputs/robocasa-flora-drawer-exact-threshold-20261103-r1/PUBLIC-JOBS.json'
jobs = json.loads(public.read_text())
records = []
for case, indices in [('000256',[0,700,750]),('000336',[0,700,750]),
                      ('000376',[0,700,750]),('000216',[0,200,250,300,750])]:
    job = next(j for j in jobs if '-case-'+case+'-' in j['id'])
    assert all(sha(Path(v['path'])) == v['sha256'] for v in job['videos'])
    reader = core.Collage([v['path'] for v in job['videos']])
    try:
        assert reader.num_frames == 751 and reader.fps == 20.
        images = [Image.fromarray(np.asarray(frame)) for frame in reader.frames(indices)]
    finally: reader.release()
    width, height = images[0].size
    assert all(image.size == (width,height) for image in images)
    sheet = Image.new('RGB',(width,(height+30)*len(images)),'white')
    draw = ImageDraw.Draw(sheet)
    for number,(control,image) in enumerate(zip(indices,images)):
        y = number*(height+30)
        sheet.paste(image,(0,y+30))
        draw.text((6,y+8),f'Case {case}, control {control}, time {control/20:.2f}s; views: left / wrist / right; human inspection only',fill='black')
    path = out/f'drawer-{case}.jpg'
    sheet.save(path,quality=95)
    records.append(dict(case=case,controls=indices,source_videos=job['videos'],
        frame_size=[width,height],sheet=str(path),sha256=sha(path),model_input=False))
receipt = dict(completed=True,source_commit=source,source_public_sha256=sha(public),
    cases=records,decoded_collages=14,seconds=time.time()-start,cuda_visible_devices='',
    model_calls=0,simulator_calls=0,policy_controls=0,optimizer_steps=0,
    limitation='Selected raw-video stills for human diagnosis; labels and sheets never enter Flora scoring.')
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
paths = sorted(out.iterdir())
manifest = dict(source_commit=source,files={p.name:sha(p) for p in paths})
archive = Path('/root/workspace/drawer-visual-audit-r1.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths: z.write(p,p.name)
    z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(completed=True,cases=4,decoded_collages=14,seconds=receipt['seconds'],
    archive=str(archive),archive_sha256=sha(archive),model_calls=0,simulator_calls=0)),flush=True)
