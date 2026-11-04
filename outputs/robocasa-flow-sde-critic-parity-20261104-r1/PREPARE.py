"""Recover exact closed-corpus critic cohorts on Linux CPU; never sample again."""
import argparse
import gc
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def tensor_sha(tensor):
    return digest(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def observation_hash(value):
    h=hashlib.sha256()
    def visit(item):
        if isinstance(item,dict):
            for key in sorted(item):h.update(str(key).encode());visit(item[key])
        elif isinstance(item,(tuple,list)):
            for child in item:visit(child)
        elif torch.is_tensor(item) or isinstance(item,np.ndarray):
            a=item.detach().cpu().numpy() if torch.is_tensor(item) else item
            h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        else:h.update(repr(item).encode())
    visit(value);return h.hexdigest()


class ClosedArchive:
    def __init__(self,root,receipt_path):
        self.root=root;self.receipt=json.loads(receipt_path.read_text())
        self.manifest_path=Path(self.receipt['manifest'])
        assert file_sha(self.manifest_path)==self.receipt['manifest_sha256']
        self.files=json.loads(self.manifest_path.read_text())['files']
        self.offsets={};offset=0
        for name,item in sorted(self.files.items()):
            member=tarfile.TarInfo(name);member.size=item['bytes'];member.mode=item['mode']
            member.mtime=item['mtime_ns']/1e9
            header=len(member.tobuf(format=tarfile.PAX_FORMAT))
            self.offsets[name]=offset+header;offset+=header+((item['bytes']+511)//512)*512
        self.stream=Path(self.receipt['archive']).open('rb');self.reads={}

    def load(self,path,qualified):
        path=Path(path);name=str(path.relative_to(self.root));item=self.files[name]
        assert dict(bytes=item['bytes'],sha256=item['sha256'])==qualified[str(path)]
        self.stream.seek(self.offsets[name]);data=self.stream.read(item['bytes'])
        assert len(data)==item['bytes'] and digest(data)==item['sha256'],name
        self.reads[name]=dict(bytes=len(data),sha256=digest(data),offset=self.offsets[name])
        if name.endswith('.gz'):data=gzip.decompress(data)
        return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    root=args.root;task='PickPlaceDrawerToCounter';out=args.output
    out.mkdir(parents=True,exist_ok=False)
    receipt_path=root/f'CORPUS-ARCHIVE-{task}-R1-RECLAIMED.json'
    archive=ClosedArchive(root,receipt_path);audits=[];references={}
    for ref in archive.receipt['full_audit_refs']:
        p=Path(ref['path']);assert file_sha(p)==ref['sha256'];references[str(p)]=ref['sha256']
        audits.extend(json.loads(p.read_text())['audits'])
    rows=[]
    for step in [0,1,10,120]:
        collection=step+1;segment='q1-r1' if step==0 else 'production-r1'
        stage=root/task/segment;cfg=stage/'resolved-config.json'
        assert json.loads((stage/'clock.json').read_text())['exit_code']==0
        candidates=[r for r in audits if r['task']==task and r['root']==str(stage)
                    and r['collection_iteration']==collection]
        assert len(candidates)==1,(step,len(candidates));audit=candidates[0]
        jp=Path(audit['actor_sampling_environment_join']);assert file_sha(jp)==audit['join_sha256']
        fp=Path(audit['raw_file_manifest']);assert file_sha(fp)==audit['raw_file_manifest_sha256']
        qualified=json.loads(fp.read_text())['files'];joins=json.loads(jp.read_text())['rows']
        markers=list((root/task).glob(f'*/{task}/checkpoints/global_step_{step}/pilot-boundary.json'))
        assert markers
        weights=[(p,json.loads(p.read_text())) for p in markers]
        model_refs=[(p,next(x for x in b['actor_checkpoint_files'] if x['path'].endswith('/full_weights.pt'))) for p,b in weights]
        assert len({r['sha256'] for _,r in model_refs})==1
        marker,model=model_refs[0]
        for actor_rank in [0,1]:
            actor_path=stage/'training-rollouts'/f'iteration-{collection:04d}-policy-{step:04d}-rank-{actor_rank:02d}.pt'
            saved=archive.load(actor_path,qualified);batch=saved['rollout_batch']
            assert saved['producing_policy_iteration']==step and not saved['evaluation_data']
            assert batch['prev_logprobs'].shape==(12,64,50,12)
            for time_index in [0,6]:
                group=sorted((r for r in joins if r['actor_record']==str(actor_path)
                    and r['time_index']==time_index and r['actor_lane']<16),key=lambda r:r['actor_lane'])
                assert [r['actor_lane'] for r in group]==list(range(16))
                assert len({r['sampling_record'] for r in group})==len({r['environment_record'] for r in group})==1
                sample=archive.load(group[0]['sampling_record'],qualified)
                environment=archive.load(group[0]['environment_record'],qualified)
                assert sample['mode']=='train' and not sample['terminal_value_only']
                assert sample['producing_policy_iteration']==step
                obs={k:v for k,v in environment['current_observation'].items() if k!='task_metadata'}
                assert observation_hash(obs)==sample['public_observation_sha256']
                lanes=torch.tensor([r['sampling_lane'] for r in group])
                fields={k:v[time_index,:16].clone() for k,v in batch['forward_inputs'].items()}
                for name,public in [('observation/image','main_images'),('observation/wrist_image','wrist_images'),
                                    ('observation/extra_view_image','extra_view_images')]:
                    image=obs[public][lanes].clone()
                    if name in fields:assert torch.equal(fields[name],image)
                    fields[name]=image
                chains=sample['forward_sampling_fields']['chains'][lanes].clone()
                if 'chains' in fields:assert torch.equal(fields['chains'],chains)
                fields['chains']=chains
                for name in ['model_action','action','denoise_inds']:
                    assert torch.equal(fields[name],sample['forward_sampling_fields'][name][lanes])
                old=batch['prev_logprobs'][time_index,:16].clone()
                values=batch['prev_values'][time_index,:16].clone()
                assert torch.equal(old,sample['old_logprobs'][lanes])
                assert torch.equal(values,sample['old_values'][lanes])
                payload=dict(forward_inputs=fields,old_logprobs=old,old_values=values,
                    returns=batch['returns'][time_index,:16].clone(),
                    loss_mask=batch['loss_mask'][time_index,:16].clone(),
                    loss_mask_sum=batch.get('loss_mask_sum',None)[time_index,:16].clone()
                        if batch.get('loss_mask_sum') is not None else None)
                name=f'policy-{step:03d}-actor-{actor_rank:02d}-chunk-{time_index:02d}.pt'
                p=out/name;torch.save(payload,p)
                row=dict(task=task,policy_step=step,collection_iteration=collection,actor_rank=actor_rank,
                    time_index=time_index,actor_lanes=list(range(16)),sampling_rank=sample['rank'],
                    sampling_ordinal=sample['prediction_ordinal'],sampling_lanes=lanes.tolist(),
                    episode_ids=[r['episode_id'] for r in group],fixture=str(p),fixture_sha256=file_sha(p),
                    field_tensor_sha256={k:tensor_sha(v) for k,v in fields.items()},
                    old_logprobs_sha256=tensor_sha(old),old_values_sha256=tensor_sha(values),
                    config=str(cfg),config_sha256=file_sha(cfg),join=str(jp),join_sha256=file_sha(jp),
                    actor_record=str(actor_path),sampling_record=group[0]['sampling_record'],
                    environment_record=group[0]['environment_record'],checkpoint=dict(model,step=step,
                        marker=str(marker),marker_sha256=file_sha(marker)),native_record_joins_exact=True)
                rows.append(row);print(json.dumps({k:row[k] for k in ['policy_step','actor_rank','time_index','sampling_rank']}),flush=True)
                write(out/'PREPARED.json',dict(completed=False,cohorts=rows))
                del sample,environment,payload,fields,chains
            del batch,saved;gc.collect()
    assert len(rows)==16
    write(out/'PREPARED.json',dict(completed=True,task=task,cohorts=rows,decisions=256,
        archive=str(archive.receipt['archive']),manifest=str(archive.manifest_path),
        manifest_sha256=file_sha(archive.manifest_path),raw_members=archive.reads,
        audit_references=references,script_sha256=file_sha(Path(__file__)),new_controls=0,SGD=0))


if __name__=='__main__':main()
