"""CPU interface check against the actual pinned VLA wrapper with a deterministic model fixture."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys

import numpy as np
import torch
from query_capture import capture_first_query, POLICY, UPSTREAM

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"upstream"))
from reground.policies.vla import VLA,chunk_seed

class Fixture:
    def predict_action_chunk(self,batch,*,noise):
        return noise[:,:,:7]+batch["signal"].to(torch.float32)

v=VLA.__new__(VLA)
v.policy=Fixture()
v.device,v.revision,v.chunk_size,v.noise_dim="cpu",POLICY,50,32
v.env_pre=v.post=v.env_post=lambda x:x
v.pre=lambda x:{**x,"signal":torch.tensor(.25,dtype=torch.bfloat16)}
seeds=[17,31,41,53]
obs=[dict(pixels=dict(image=np.full((8,8,3),i,np.uint8),image2=np.full((8,8,3),i+4,np.uint8))) for i in range(4)]
tasks=["fixture"]*4
def run(q): return v.predict_chunk(obs,tasks,seeds,q)
a,b=run(0),run(1)
with TemporaryDirectory(dir=here) as tmp:
    to=Path(tmp)/"case"
    with capture_first_query(v,to,[(0,"fixture",i,0,i) for i in range(4)],lambda:{"fixture":True},UPSTREAM):
        assert np.array_equal(a,run(0)) and np.array_equal(b,run(1))
    d=json.loads((to/"receipt.json").read_text())
    with np.load(to/"arrays.npz") as z:
        assert z[d["proposal"]["array"]].tobytes()==a.tobytes()
        assert z[d["noise"]["array"]].tobytes()==v._noise(seeds,0).numpy().tobytes()
        sig=d["processed"]["signal"]
        assert sig["dtype"]=="torch.bfloat16" and sig["shape"]==[]
        assert z[sig["array"]].tobytes()==torch.tensor([.25],dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    print(json.dumps(dict(native_wrapper="reground.policies.vla.VLA",model="CPU fixture",cuda_visible=os.getenv("CUDA_VISIBLE_DEVICES"),
                          zero_and_next_query_passthrough=True,raw_proposal_and_native_noise_recovered=True,bfloat16_scalar_exact=True,
                          chunk_seeds=[chunk_seed(s,0) for s in seeds])))
