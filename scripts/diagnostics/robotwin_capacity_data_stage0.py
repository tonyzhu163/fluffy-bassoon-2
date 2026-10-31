"""Bounded allocated-CPU inventory of the exact phone demonstration archive.

Downloads data only. No model/simulator imports, action labels, or GPU work.
Unknown success/alignment metadata stays unknown, never a manufactured teacher.
"""
import argparse
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile

REV='3dc3b798668feb99ac61cc9086d84cbcc3d79186'
SHA='745a98306a28fd5e112388489b1973c43b9bb9946233c7ab4f1e40c6d1b5d065'
SIZE=255542392
URL=f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/dataset/place_phone_stand/aloha-agilex_clean_50.zip'
ROBOTWIN='dca9ec682688821c42944eacd7f5dd9bfb15396f'
BLOCKED=('torch','jax','sapien','rlinf','openpi','fluffy_bassoon.rl')


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in BLOCKED):
            raise ImportError(f'Stage0 rejects model/simulator module {fullname}')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4<<20),b''): digest.update(block)
    return digest.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('x') as target:
        json.dump(value,target,indent=2,allow_nan=False);target.write('\n');target.flush();os.fsync(target.fileno())
    temporary.replace(path)


def scalar(value):
    if isinstance(value,bytes): return value.decode('utf-8',errors='replace')
    if hasattr(value,'tolist'): value=value.tolist()
    if isinstance(value,(list,tuple)): return [scalar(v) for v in value[:16]]
    if isinstance(value,dict): return {str(k):scalar(v) for k,v in value.items()}
    if isinstance(value,float) and not math.isfinite(value): return dict(nonfinite_metadata=str(value))
    if value is None or isinstance(value,(str,int,float,bool)): return value
    return str(value)


def ordered_hdf5(bundle):
    names=[r.filename for r in bundle.infolist()]
    if len(names)!=len(set(names)): raise ValueError('Duplicate ZIP member names')
    rows=[]
    for info in bundle.infolist():
        name=PurePosixPath(info.filename)
        if name.is_absolute() or '..' in name.parts or ((info.external_attr>>16)&0o170000)==0o120000:
            raise ValueError('Unsafe archive member path/symlink')
        if name.suffix.lower() not in ('.hdf5','.h5'): continue
        match=re.fullmatch(r'episode[_-]?(\d+)',name.stem)
        if match is None: raise ValueError('Require unambiguous canonical episode index, no lexical ordering')
        rows.append((int(match[1]),info))
    if len(rows)<12 or len({index for index,_ in rows})!=len(rows):
        raise ValueError('Need at least twelve distinct whole source episodes')
    return sorted(rows,key=lambda row:row[0])


def plain_metadata(bundle,candidates):
    """Bind actual seed/scene/instruction sidecars. Pickles are never opened."""
    root=PurePosixPath(candidates[0][1].filename).parent.parent
    seed_name=str(root/'seed.txt');scene_name=str(root/'scene_info.json')
    if bundle.getinfo(seed_name).file_size>65536 or bundle.getinfo(scene_name).file_size>2<<20:
        raise ValueError('Unbounded plain metadata sidecar before decompression')
    seed_bytes=bundle.read(seed_name);scene_bytes=bundle.read(scene_name)
    if len(seed_bytes)>65536 or len(scene_bytes)>2<<20: raise ValueError('Unbounded plain metadata sidecar')
    seeds=[int(value) for value in seed_bytes.decode().split()]
    scene=json.loads(scene_bytes)
    if len(seeds)<len(candidates) or not isinstance(scene,dict): raise ValueError('Missing actual canonical seed/scene map')
    return dict(root=str(root),seeds=seeds,scene=scene,refs=dict(
        seeds=dict(member=seed_name,sha256=hashlib.sha256(seed_bytes).hexdigest()),
        scenes=dict(member=scene_name,sha256=hashlib.sha256(scene_bytes).hexdigest())))


def episode_sidecars(bundle,metadata,index):
    name=str(PurePosixPath(metadata['root'])/'instructions'/f'episode{index}.json')
    info=bundle.getinfo(name)
    if info.file_size>256<<10: raise ValueError('Unbounded instruction metadata')
    payload=bundle.read(info);instructions=json.loads(payload)
    seen=instructions.get('seen')
    if not isinstance(seen,list) or not seen or not isinstance(seen[0],str) or not seen[0].strip():
        raise ValueError('Require an actual source instruction from the published seen list')
    key=f'episode_{index}'
    if key not in metadata['scene'] or index>=len(metadata['seeds']): raise ValueError('Episode has no actual reset/scene binding')
    return dict(reset_seed=metadata['seeds'][index],scene_metadata=metadata['scene'][key],
        selected_instruction=seen[0],instruction_selection='first published seen instruction, fixed before fitting',
        instruction_ref=dict(member=name,sha256=hashlib.sha256(payload).hexdigest()),
        seed_scene_refs=metadata['refs'])


def select_whole_episodes(episodes):
    successes=[r for r in episodes if r['explicit_success'] is True]
    unknown=[r for r in episodes if r['explicit_success'] is None]
    if len(successes)>=12:
        chosen=successes[:12];status='explicit_success_metadata'
    elif unknown:
        chosen=[r for r in episodes if r['explicit_success'] is not False][:12]
        status='canonical_expert_candidates_split_pending_author_success_and_alignment_review'
    else:
        chosen=[];status='insufficient_explicit_successes_no_teacher_split'
    return dict(status=status,train=[r['canonical_index'] for r in chosen[:8]],
        validation=[r['canonical_index'] for r in chosen[8:12]],
        ordered_candidate_indices=[r['canonical_index'] for r in episodes],
        explicit_success_indices=[r['canonical_index'] for r in successes],
        unknown_success_indices=[r['canonical_index'] for r in unknown],whole_episode_split=True,
        command_state_alignment_qualified=False)


def hdf5_inventory(path,guard):
    import h5py
    import numpy as np
    result=dict(attributes={},datasets={},explicit_success=None,seed_metadata={},instruction_metadata={})
    with h5py.File(path,'r') as data:
        result['attributes']={str(k):scalar(v) for k,v in data.attrs.items()}
        def visit(name,obj):
            guard()
            if not isinstance(obj,h5py.Dataset): return
            row=dict(shape=list(obj.shape),dtype=str(obj.dtype),attributes={str(k):scalar(v) for k,v in obj.attrs.items()})
            lowered=name.lower()
            if obj.ndim==2 and obj.shape[1]==14 and obj.dtype.kind in 'fi':
                values=np.asarray(obj)
                if not np.isfinite(values).all(): raise ValueError('Nonfinite physical14D data')
                row.update(first=values[0].tolist(),last=values[-1].tolist(),minimum=values.min(0).tolist(),maximum=values.max(0).tolist())
                row['data_semantics']='not inferred from numeric range; requires author-source alignment'
            if any(key in lowered for key in ('success','seed','instruction','prompt','time','timestamp','frequency','fps')):
                if obj.size<=4096:
                    values=np.asarray(obj);row['metadata_values']=scalar(values)
                elif obj.ndim==1:
                    row['first_last']=[scalar(obj[0]),scalar(obj[-1])]
            result['datasets'][name]=row
        data.visititems(visit)
        for key,value in data.attrs.items():
            name=key.lower()
            if name in ('success','episode_success','success_once'):
                val=scalar(value)
                if isinstance(val,(bool,int)) and val in (0,1): result['explicit_success']=bool(val)
            if name in ('seed','env_seed','reset_seed','episode_seed','ep_num'): result['seed_metadata'][key]=scalar(value)
            if any(x in name for x in ('instruction','prompt')): result['instruction_metadata'][key]=scalar(value)
        for name in ('success','episode_success','success_once'):
            if name in data and isinstance(data[name],h5py.Dataset):
                values=np.asarray(data[name])
                if values.size and np.isin(values,(0,1)).all(): result['explicit_success']=bool(values.reshape(-1)[-1])
    return result


def author_sources(root,guard):
    root=Path(root).resolve()
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,timeout=10).strip()
    if commit!=ROBOTWIN: raise ValueError('Author writer/executor source is not the pinned deployed RoboTwin')
    names=subprocess.check_output(['git','-C',str(root),'ls-files','*.py'],text=True,timeout=10).splitlines()
    rows=[]
    for name in names:
        if not (name.startswith(('envs/','script/','scripts/','policy/pi0/','robotwin/envs/'))
                or 'collect' in PurePosixPath(name).stem): continue
        guard();path=root/name
        if path.is_symlink() or path.stat().st_size>512000: continue
        text=path.read_text(errors='replace')
        if not any(word in text for word in ('h5py','joint_action','create_dataset')): continue
        lines=text.splitlines();indices=set()
        for index,line in enumerate(lines):
            if any(word in line for word in ('joint_action','create_dataset','h5py','save_data','check_success','get_left_arm_jointState','get_right_arm_jointState')):
                indices.update(range(max(0,index-3),min(len(lines),index+5)))
        excerpts=[dict(line=i+1,text=lines[i]) for i in sorted(indices)[:100]]
        rows.append(dict(path=str(path),relative_path=name,sha256=sha(path),excerpts=excerpts))
        if len(rows)==16: break
    return dict(commit=commit,files=rows,alignment_status='source excerpts for explicit review; no automatic state/command inference')


def a1_grippers(roots,guard):
    """Immutable completed decision NPZs only; no images, motor or live ledger."""
    import numpy as np
    values=[];issued=[];refs=[]
    for root in roots:
        root=Path(root).resolve()
        for path in sorted((root/'episodes').glob('*/chunks/*.json')):
            guard();row=json.loads(path.read_text())
            if row['status']!='complete': raise ValueError('A1 gripper audit refuses unresolved chunk intents')
            decision=row['decision_arrays'];native=row['native_result']['arrays']
            files=[]
            for ref in (decision,native):
                target=(root/ref['path']).resolve()
                if not target.is_relative_to(root) or sha(target)!=ref['sha256']:
                    raise ValueError('A1 gripper audit array SHA/containment mismatch')
                files.append(target)
            with np.load(files[0],allow_pickle=False) as data,np.load(files[1],allow_pickle=False) as feedback:
                actions=data['issued_motor_joint_targets'];native_mask=feedback['valid_action_mask'].reshape(-1)
                if not np.isin(native_mask,(0,1)).all(): raise ValueError('Nonbinary native execution mask')
                mask=native_mask.astype(bool)
                if actions.ndim!=2 or actions.shape[1]!=14 or len(mask)>len(actions):
                    raise ValueError('A1 actual command layout or native mask changed')
                current=actions[:len(mask)][mask][:,[6,13]]
                if len(current)!=row['native_result']['executed_primitive_steps']:
                    raise ValueError('A1 gripper executed duration differs from actual native mask')
                values.append(current.copy())
                issued.append(actions[:,[6,13]].copy())
            refs.append(dict(path=str(path),sha256=sha(path)))
    if not values: raise ValueError('Requested immutable A1 gripper audit had no complete chunks')
    def describe(parts):
        joined=np.concatenate(parts)
        finite=np.isfinite(joined)
        columns=[joined[finite[:,i],i] for i in range(2)]
        return dict(slots=len(joined),finite_counts=finite.sum(0).tolist(),nonfinite_counts=(~finite).sum(0).tolist(),
            minimum=[float(x.min()) if len(x) else None for x in columns],
            maximum=[float(x.max()) if len(x) else None for x in columns],
            quantiles=[[float(np.quantile(x,q)) if len(x) else None for x in columns] for q in (.01,.5,.99)],
            outside_0_1_counts=(((joined<0)|(joined>1))&finite).sum(0).tolist())
    return dict(semantics='pre-actuator decoded native absolute gripper commands; clipping occurs downstream',
        indices=[6,13],total_issued=describe(issued),actual_valid=describe(values),chunk_refs=refs)


def main(argv=None):
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase-root',type=Path,required=True)
    p.add_argument('--robotwin-root',type=Path,required=True)
    p.add_argument('--archive-cache',type=Path)
    p.add_argument('--allocation',type=Path,required=True)
    p.add_argument('--allocation-sha256',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--a1-corpus',type=Path,action='append',default=[])
    p.add_argument('--max-seconds',type=int,default=240)
    args=p.parse_args(argv)
    if sys.platform!='linux' or not 1<=args.max_seconds<=240: raise RuntimeError('Actual allocated Linux CPU only')
    if any(name==bad or name.startswith(bad+'.') for name in sys.modules for bad in BLOCKED):
        raise RuntimeError('Model/simulator module was already loaded in data worker')
    sys.meta_path.insert(0,NoModels())
    if sha(args.allocation)!=args.allocation_sha256: raise ValueError('Actual CPU ticket SHA mismatch')
    ticket=json.loads(args.allocation.read_text())
    cpu_cap=ticket.get('maximum_cpu_worker_seconds')
    if (ticket.get('hostname')!=socket.gethostname() or ticket.get('gpu_count')!=0
            or type(cpu_cap) is not int or not 1<=cpu_cap<=300
            or ticket.get('source_commit')!=args.source_commit
            or Path(ticket.get('output','/')).resolve()!=args.output.resolve()
            or ticket.get('phase_id')!='frozen-pi05-endtoend-20261031-r1'):
        raise ValueError('Require actual zero-GPU <=300CPU-worker-s original-phase admission')
    project=Path(__file__).resolve().parents[2]
    if (subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True,timeout=10).strip()!=args.source_commit
            or subprocess.check_output(['git','-C',str(project),'status','--porcelain','--untracked-files=no'],text=True,timeout=10).strip()):
        raise ValueError('Require the exact clean Git-deployed data-stage code')
    phase=args.phase_root.resolve();output=args.output.resolve()
    if not output.is_relative_to(phase) or output==phase: raise ValueError('Keep new data inside original phase output root')
    def guard():
        if time.monotonic()-started>=args.max_seconds: raise TimeoutError('Original bounded data preparation clock expired')
        if not phase.exists(): raise ValueError('Missing actual original phase root')
    external_archive_bytes=(SIZE if args.archive_cache and not args.archive_cache.resolve().is_relative_to(phase) else 0)
    def storage():
        guard();total=0
        for directory,dirs,files in os.walk(phase,followlinks=False):
            if any((Path(directory)/name).is_symlink() for name in (*dirs,*files)): raise ValueError('Phase output symlink rejected')
            total+=sum((Path(directory)/name).stat().st_size for name in files)
        total+=external_archive_bytes
        if total+(1<<30)>64<<30: raise RuntimeError('Original64GiB reserve cannot admit this data stage')
        return total
    storage();output.mkdir(parents=True,exist_ok=False)
    result=dict(format='robotwin_capacity_stage0_v1',status='running',source_only=True,model_readiness=False,
        allocation_sha256=args.allocation_sha256,hostname=socket.gethostname(),dataset_revision=REV,
        robot_actions=0,model_imports=[],GPU_seconds=0,episodes=[])
    try:
        sources=author_sources(args.robotwin_root,guard);write(output/'author-source-excerpts.json',sources)
        if args.archive_cache:
            cached=args.archive_cache.resolve()
            if cached.stat().st_size!=SIZE or sha(cached)!=SHA: raise ValueError('Existing data input hash/size mismatch')
            archive=output/'phone-aloha-agilex-clean50.zip'
            with cached.open('rb') as source,archive.open('xb') as target:
                for block in iter(lambda:source.read(1<<20),b''): guard();target.write(block)
            result['original_cache']=dict(path=str(cached),sha256=SHA,bytes=SIZE,
                external_bytes_also_charged_to_phase=external_archive_bytes)
        else:
            archive=output/'phone-aloha-agilex-clean50.zip';partial=archive.with_suffix('.zip.partial')
            with urllib.request.urlopen(URL,timeout=min(30,args.max_seconds)) as response,partial.open('xb') as target:
                downloaded=0
                for block in iter(lambda:response.read(1<<20),b''):
                    guard();downloaded+=len(block)
                    if downloaded>SIZE: raise ValueError('Unexpected archive size; no replacement/mirror fallback')
                    target.write(block)
                target.flush();os.fsync(target.fileno())
            partial.replace(archive)
        guard()
        if archive.stat().st_size!=SIZE or sha(archive)!=SHA: raise ValueError('Pinned public archive byte/hash mismatch')
        result['archive']=dict(path=str(archive),sha256=SHA,bytes=SIZE,url=URL)
        with zipfile.ZipFile(archive) as bundle:
            candidates=ordered_hdf5(bundle)
            metadata=plain_metadata(bundle,candidates)
            result['plain_metadata_refs']=metadata['refs']
            result['archive_hdf5_indices']=[index for index,_ in candidates]
            result['archive_members']=[dict(path=info.filename,bytes=info.file_size,crc32=info.CRC) for info in bundle.infolist()]
            data_root=output/'whole-demos';data_root.mkdir()
            selected=[];unknown=[];inspected_bytes=0
            for index,info in candidates:
                guard();path=data_root/f'episode{index}.hdf5'
                inspected_bytes+=info.file_size
                if info.file_size>512<<20 or inspected_bytes>8<<30:
                    raise ValueError('Whole episode inspection exceeds its bounded raw-data reserve')
                if storage()+info.file_size+(256<<20)>64<<30:
                    raise RuntimeError('Original aggregate storage cannot admit the next whole demo')
                with bundle.open(info) as source,path.open('xb') as target:
                    for block in iter(lambda:source.read(1<<20),b''): guard();target.write(block)
                inventory=hdf5_inventory(path,guard)
                row=dict(canonical_index=index,member=info.filename,path=str(path),bytes=path.stat().st_size,
                         sha256=sha(path),**inventory,**episode_sidecars(bundle,metadata,index))
                result['episodes'].append(row)
                if inventory['explicit_success'] is True: selected.append(row)
                elif inventory['explicit_success'] is None: unknown.append(row)
                if len(selected)>=12: break
                # Lack of success metadata is a real qualification question,
                # not permission to decompress all50 or invent twelve successes.
                if len(result['episodes'])>=12 and unknown: break
            result['selection']=select_whole_episodes(result['episodes'])
        result.update(status='inventory_complete_alignment_review_required',storage_bytes=storage(),
                      elapsed_seconds=time.monotonic()-started,hdf5_import_only=True,model_readiness=False)
        if args.a1_corpus:
            if any(not root.resolve().is_relative_to(phase) for root in args.a1_corpus):
                raise ValueError('Audit only this original phase\'s immutable A1 corpora')
            statistics=a1_grippers(args.a1_corpus,guard);write(output/'a1-gripper-statistics.json',statistics)
            result['gripper_statistics']=dict(path=str(output/'a1-gripper-statistics.json'),sha256=sha(output/'a1-gripper-statistics.json'))
        write(output/'inventory.json',result)
    except BaseException as error:
        result.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)),elapsed_seconds=time.monotonic()-started)
        write(output/'failure.json',result);raise


if __name__=='__main__': main()
