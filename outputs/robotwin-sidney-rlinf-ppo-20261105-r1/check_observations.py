"""Read-only CPU inspection inside the existing allocated training container."""
import hashlib
import json
from pathlib import Path
import socket
import sys

from PIL import Image
import torch


root=Path(sys.argv[1]);probe=json.loads((root/'gpu-probe.json').read_text())
assert socket.gethostname()==probe['hostname'],'Inspect only the owning allocation'
torch.set_num_threads(1)
rows=[];images=root/'observation-check';images.mkdir(exist_ok=True)
for rank in range(4):
    path=root/'initial-states'/f'source-rank-{rank:02d}-0000.pt'
    value=torch.load(path,map_location='cpu',weights_only=False)
    raw=value['initial_observation'];cameras={}
    for key in ('main_images','wrist_images'):
        tensor=raw[key];assert tensor.dtype==torch.uint8 and tensor.shape[-1]==3
        frames=tensor.reshape(-1,*tensor.shape[-3:])
        sampled=frames[:,::16,::16].float().flatten(1)
        std=sampled.std(dim=1);span=sampled.amax(dim=1)-sampled.amin(dim=1)
        assert bool((std>1).all() and (span>5).all()),f'Blank/constant {key} on rank{rank}'
        cameras[key]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),image_count=len(frames),
            minimum_sampled_image_std=float(std.min()),maximum_sampled_image_std=float(std.max()),
            minimum_sampled_image_range=float(span.min()))
    assert bool(torch.isfinite(raw['states']).all())
    bindings=json.loads((root/'scene-bindings'/f'rank-{rank:02d}-00000.json').read_text())
    assert len(bindings)==len(raw['task_descriptions'])==64
    assert all(b['instruction']==instruction for b,instruction in zip(bindings,raw['task_descriptions']))
    main=raw['main_images'][0];wrists=raw['wrist_images'][0]
    frames=[main,*wrists];height=max(t.shape[0] for t in frames);width=sum(t.shape[1] for t in frames)
    montage=Image.new('RGB',(width,height));x=0
    for frame in frames:
        montage.paste(Image.fromarray(frame.numpy()),(x,0));x+=frame.shape[1]
    montage.save(images/f'rank-{rank:02d}.jpg',quality=85)
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):digest.update(block)
    rows.append(dict(rank=rank,path=str(path),sha256=digest.hexdigest(),cameras=cameras,
        scene_instruction_binding_count=len(bindings),first_scene_binding=bindings[0]))
assert not torch.cuda.is_initialized(),'Observation inspection must remain CPU-only'
report=dict(status='complete',hostname=socket.gethostname(),cuda_initialized=False,
    observation_count=256,camera_image_count=sum(c['image_count'] for row in rows for c in row['cameras'].values()),
    weights_or_contract_changed=False,rows=rows)
(root/'observation-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
