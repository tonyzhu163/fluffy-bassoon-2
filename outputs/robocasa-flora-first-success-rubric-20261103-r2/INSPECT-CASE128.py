"""Decode five retained multi-view frames for human diagnosis only, on Linux CPU."""
import json
import os
from pathlib import Path
import platform
import sys

assert platform.system() == 'Linux'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image, ImageDraw

root = Path('/mowenjie/fluffy-bassoon-2')
out = root/'outputs/robocasa-flora-first-success-rubric-20261103-r2'
sys.path.insert(0, str(root/'deps/nvidia-flora-129a2ada-20261101-r1/repo'))
from rewarder.batch_utils import core
jobs = json.loads((out/'PUBLIC-JOBS.json').read_text())
job = next(row for row in jobs if '-case-000128-' in row['id'])
reader = core.Collage([video['path'] for video in job['videos']])
indices = [0, 350, 400, 450, 750]
try:
    images = [Image.fromarray(np.asarray(frame)) for frame in reader.frames(indices)]
finally:
    reader.release()
width, height = images[0].size
sheet = Image.new('RGB', (width, (height+24)*len(images)), 'white')
draw = ImageDraw.Draw(sheet)
for number, (control, image) in enumerate(zip(indices, images)):
    sheet.paste(image, (0, number*(height+24)+24))
    draw.text((6, number*(height+24)+5),
        f'Case128: control {control}, time {control/20:.2f}s; native first success384', fill='black')
sheet.save(out/'drawer128-prefix-inspection.jpg', quality=95)
