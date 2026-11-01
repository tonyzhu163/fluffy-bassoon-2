import hashlib,json,os,time
from pathlib import Path,PurePosixPath
os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from huggingface_hub import HfApi,snapshot_download
revision='031299e46e452fc0cedb4e0975eb9fa95fc37783';repo='nvidia/Flora-Rewarder'
root=Path('/mowenjie/fluffy-bassoon-2/inputs/flora-rewarder-031299e4-20261101-r1');receipt=root.parent/(root.name+'-receipt');receipt.mkdir(exist_ok=False)
assert not root.exists(),'Fresh pinned model directory required'
started=time.time();api=HfApi(token=False);info=api.model_info(repo,revision=revision,files_metadata=True,token=False);assert info.sha==revision
metadata=dict(repo_id=repo,sha=info.sha,files=[dict(path=x.rfilename,size=x.size,blob_id=x.blob_id,lfs=x.lfs) for x in info.siblings]);(receipt/'HF-METADATA.json').write_text(json.dumps(metadata,indent=2,default=lambda x:vars(x))+'\n')
for row in info.siblings:
 q=PurePosixPath(row.rfilename);assert not q.is_absolute() and '..' not in q.parts
print(json.dumps(dict(event='ready40_public_download_start',repo=repo,revision=revision,files=len(info.siblings),token=False,model_directory=str(root))),flush=True)
try:
 snapshot_download(repo_id=repo,revision=revision,local_dir=str(root),token=False,max_workers=4)
 files=[]
 for row in info.siblings:
  p=root/row.rfilename;assert p.is_file();size=p.stat().st_size
  if row.size is not None:assert size==row.size
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=str(p),relative_path=row.rfilename,bytes=size,sha256=digest))
 assert sum(x['bytes'] for x in files)<64<<30
 result=dict(completed=True,repo_id=repo,model_revision=revision,model_directory=str(root),files=files,bytes=sum(x['bytes'] for x in files),started_unix=started,ended_unix=time.time(),token=False,model_loaded=False,arrays_deserialized=False,gpu_allocated=False,official_pinned_Flora_asset_only=True)
 (receipt/'READY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='ready40_public_download_verified',bytes=result['bytes'],files=len(files),receipt=str(receipt/'READY.json'))),flush=True)
except Exception as e:
 (receipt/'FAILURE.json').write_text(json.dumps(dict(completed=False,exception_type=type(e).__name__,started_unix=started,failed_unix=time.time(),partials_preserved=True),indent=2)+'\n');print('FAILED_TYPE='+type(e).__name__,flush=True);raise SystemExit(1)
