"""Whole-demo reader and supervised primitives for the bounded reactive probe.

No downloads, scheduler, simulator or learner is constructed on import. Archive
schema/alignment are explicit, SHA-bound qualification inputs rather than guesses.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import zipfile

import numpy as np


PHONE_ARCHIVE_SHA = "745a98306a28fd5e112388489b1973c43b9bb9946233c7ab4f1e40c6d1b5d065"
PHONE_ARCHIVE_BYTES = 255542392
DATASET_REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
FORMAT = "robotwin_whole_demo_reactive_capacity_v1"


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8<<20),b''): value.update(block)
    return value.hexdigest()


def bound_json(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or file_sha(path)!=ref['sha256']:
        raise ValueError('Require an absolute SHA-bound immutable input')
    return json.loads(path.read_text())


def validate_demo_manifest(manifest):
    if (manifest.get('format')!=FORMAT or manifest.get('task')!='place_phone_stand'
            or manifest.get('dataset_revision')!=DATASET_REVISION
            or manifest.get('archive_sha256')!=PHONE_ARCHIVE_SHA
            or manifest.get('archive_bytes')!=PHONE_ARCHIVE_BYTES):
        raise ValueError('Bind the exact original source phone clean50 archive')
    rows=manifest['episodes']
    if (len(rows)!=12 or len({r['id'] for r in rows})!=12
            or len({r['member'] for r in rows})!=12
            or [r['canonical_index'] for r in rows]!=list(range(12))
            or [r['split'] for r in rows]!=['train']*8+['validation']*4
            or sum(r['split']=='train' for r in rows)!=8 or sum(r['split']=='validation' for r in rows)!=4):
        raise ValueError('Split exactly eight/four distinct WHOLE source episodes before windows')
    for row in rows:
        member=PurePosixPath(row['member'])
        if (member.is_absolute() or '..' in member.parts or not member.name.endswith(('.hdf5','.h5'))
                or type(row['reset_seed']) is not int or row['reset_seed']<0
                or not isinstance(row['instruction'],str) or not row['instruction'].strip()
                or row.get('success_provenance')!='author_success_filtered_collection'
                or row.get('explicit_success') is not None):
            raise ValueError('Require source-bound filtered demo identity; do not invent HDF5 success flags')
    schema=manifest['schema']
    if (schema.get('state_units')!='aloha_physical_drive_target14'
            or schema.get('action_units')!='aloha_absolute_joint_target14'
            or schema.get('camera_order')!=['head','left_wrist','right_wrist']
            or schema.get('image_encoding')!='jpeg'
            or schema.get('image_decoder')!='opencv_imdecode_color_as_native_channels'
            or len(schema.get('images',[]))!=3):
        raise ValueError('Require qualified camera and physical state/action conventions')
    offset=schema['action_offset']
    if type(offset) is not int or offset not in (0,1):
        raise ValueError('Declare the actual author-aligned command offset, not a guessed shift')
    if schema['state']==schema['actions'] and schema.get('target_kind')!='author_recorded_drive_target':
        raise ValueError('Shared state/target field needs the actual author drive-target convention')
    if schema.get('target_kind') not in ('submitted_joint_target','author_recorded_drive_target'):
        raise ValueError('Bind actual target semantics separately from measured observations')
    evidence=bound_json(manifest['alignment_qualification'])
    if (evidence.get('status')!='author_convention_verified_runtime_pending'
            or evidence.get('dataset_revision')!=DATASET_REVISION
            or evidence.get('schema')!=schema or evidence.get('source_only') is not True
            or evidence.get('robotwin_commit')!='dca9ec682688821c42944eacd7f5dd9bfb15396f'
            or evidence.get('offset_meaning')!='same-frame author-recorded drive target, followed by future frames'
            or offset!=0 or not evidence.get('source_files')):
        raise ValueError('Bind reviewed author convention; actual transform/replay qualification remains pending')
    if (manifest.get('source_inventory_sha256')!=evidence.get('source_inventory_sha256')
            or manifest.get('archive_sha256')!=evidence.get('archive_sha256')):
        raise ValueError('Author review does not bind this exact archive/inventory')
    return deepcopy(manifest)


def prepare_demo_manifest(inventory,inventory_sha,author,author_ref,*,motor_sha,norm_sha):
    """Metadata composition only. No HDF5/model/teacher result is synthesized."""
    if (inventory.get('dataset_revision')!=DATASET_REVISION or inventory['archive']['sha256']!=PHONE_ARCHIVE_SHA
            or inventory['archive']['bytes']!=PHONE_ARCHIVE_BYTES or len(inventory['episodes'])!=12
            or author['source_inventory_sha256']!=inventory_sha):
        raise ValueError('Require the exact allocated stage0 archive/inventory')
    rows=[];paths={}
    for i,row in enumerate(inventory['episodes']):
        if row['canonical_index']!=i or row['explicit_success'] is not None:
            raise ValueError('Keep actual canonical identities and absent explicit flags')
        shape=row['datasets']['joint_action/vector']['shape']
        if len(shape)!=2 or shape[1]!=14:raise ValueError('Whole physical14D episode shape changed')
        ident=f'phone-{i:02d}'
        rows.append(dict(id=ident,canonical_index=i,member=row['member'],split='train' if i<8 else 'validation',
            reset_seed=row['reset_seed'],instruction=row['selected_instruction'],
            instruction_selection=row['instruction_selection'],instruction_ref=row['instruction_ref'],
            scene_metadata=row['scene_metadata'],seed_scene_refs=row['seed_scene_refs'],
            qualified_state_count=shape[0],qualified_action_count=shape[0],explicit_success=None,
            success_provenance='author_success_filtered_collection'))
        paths[ident]={k:row[k] for k in ('path','sha256','bytes')}
    result=dict(format=FORMAT,task='place_phone_stand',dataset_revision=DATASET_REVISION,
        archive_sha256=PHONE_ARCHIVE_SHA,archive_bytes=PHONE_ARCHIVE_BYTES,source_inventory_sha256=inventory_sha,
        schema=author['schema'],alignment_qualification=author_ref,episodes=rows,actual_whole_files=paths,
        motor_sha256=motor_sha,normalization_sha256=norm_sha,
        readiness='author convention bound; official transform roundtrip and physical replay pending')
    return result


def extract_selected(archive,manifest,destination,*,guard):
    """Preserve whole chosen HDF5 members; never extract an arbitrary zip tree."""
    destination=Path(destination)
    if Path(archive).stat().st_size!=manifest['archive_bytes'] or file_sha(archive)!=manifest['archive_sha256']:
        raise ValueError('Published archive byte identity differs')
    destination.mkdir(parents=True,exist_ok=False)
    paths={}
    with zipfile.ZipFile(archive) as bundle:
        names=[row.filename for row in bundle.infolist()]
        if len(names)!=len(set(names)): raise ValueError('Ambiguous duplicate archive members')
        selected=[bundle.getinfo(row['member']) for row in manifest['episodes']]
        if sum(row.file_size for row in selected)>8<<30:
            raise ValueError('Selected full demos exceed the bounded raw-data reserve')
        for ordinal,(row,info) in enumerate(zip(manifest['episodes'],selected)):
            guard()
            if info.is_dir() or ((info.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Only actual regular HDF5 member bytes are admitted')
            path=destination/f'episode-{ordinal:02d}.hdf5'
            with bundle.open(info) as source,path.open('xb') as target:
                for block in iter(lambda:source.read(1<<20),b''):
                    guard();target.write(block)
            if path.stat().st_size!=info.file_size: raise ValueError('Incomplete demo extraction')
            paths[row['id']]=dict(path=str(path),sha256=file_sha(path),archive_member=row['member'])
    return paths


def read_episode(path,row,schema):
    import h5py
    with h5py.File(path,'r') as data:
        state=np.asarray(data[schema['state']],dtype=np.float32)
        actions=np.asarray(data[schema['actions']],dtype=np.float32)
        if (state.ndim!=2 or state.shape[1]!=14 or actions.ndim!=2 or actions.shape[1]!=14
                or not np.isfinite(state).all() or not np.isfinite(actions).all()
                or len(state)<2 or len(actions)<=schema['action_offset']
                or any(len(data[key])!=len(state) for key in schema['images'])):
            raise ValueError('Actual whole demo array shape/time alignment differs from qualification')
        if len(state)!=len(actions):raise ValueError('Whole author state/target timelines must have equal length')
        length=len(actions)-schema['action_offset']
        if length!=row['qualified_action_count'] or len(state)!=row['qualified_state_count']:
            raise ValueError('Whole episode length differs; no truncation or schema fallback')
        targets=actions[schema['action_offset']:schema['action_offset']+length].copy()
    return state,targets


def raw_observation(path,row,schema,index,state):
    import h5py
    import cv2
    views=[]
    with h5py.File(path,'r') as data:
        for key in schema['images']:
            value=data[key][index]
            if schema['image_encoding']=='jpeg':
                payload=value.tobytes() if hasattr(value,'tobytes') else bytes(value)
                # Author camera returns RGB, but images_encoding passes that
                # array directly to OpenCV's BGR-oriented JPEG writer. Its
                # matching inverse restores the original native channel order;
                # PIL's nominal RGB decode would swap the red/blue channels.
                image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
            else:
                image=np.asarray(value)
            if image is None or image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
                raise ValueError('Require actual qualified HWC uint8 RGB views')
            views.append(image.copy())
    if views[1].shape!=views[2].shape: raise ValueError('Wrist cameras have different declared shapes')
    return dict(main_images=views[0][None],wrist_images=np.stack(views[1:])[None],
                states=state[index:index+1].copy(),task_descriptions=[row['instruction']])


def whole_episode_chunks(targets,*,horizon=32):
    """Nonoverlapping anchors retain every target once; only padding is masked."""
    if targets.ndim!=2 or targets.shape[1]!=14 or len(targets)<1:
        raise ValueError('Require nonempty complete physical action targets')
    for index in range(0,len(targets),horizon):
        valid=min(horizon,len(targets)-index)
        padded=np.repeat(targets[index+valid-1:index+valid],horizon,axis=0)
        padded[:valid]=targets[index:index+valid]
        yield index,padded,np.arange(horizon)<valid


def transformed_target(model,raw,target,*,roundtrip_atol=1e-5):
    """Use the SAME loaded checkpoint input/output transforms, including prompt."""
    import torch
    batch=dict(**{'observation/image':raw['main_images'],'observation/wrist_image':raw['wrist_images'],
                  'observation/state':raw['states']},actions=target[None],prompt=raw['task_descriptions'])
    processed=model.input_transform(batch,transpose=False)
    actions=model._actions_to_device(processed['actions'])
    observation=model._observation_dict_to_device(processed)
    if tuple(actions.shape)!=(1,32,32): raise ValueError('Teacher target must be native normalized padded32D/H32')
    decoded=model.decode_actions(actions,observation.state).detach().cpu().numpy()[0]
    if not np.allclose(decoded,target,rtol=1e-5,atol=roundtrip_atol):
        raise ValueError('Official inverse/delta/Normalize/padding target failed physical14D roundtrip')
    return observation,actions,dict(max_abs_roundtrip_error=float(np.max(np.abs(decoded-target))))


def masked_mse(prediction,target,valid):
    import torch
    if prediction.shape!=target.shape or valid.shape!=prediction.shape[:2] or valid.dtype!=torch.bool:
        raise ValueError('Require an explicit valid-time mask; no padded target loss')
    denominator=valid.sum()*prediction.shape[-1]
    if denominator<=0: raise ValueError('No actual action targets in this batch')
    return ((prediction-target).square()*valid[...,None]).sum()/denominator


def reactive_flow_loss(model,adapter,batch,*,rng,generator,fixed_time=None,fixed_noise=None,
                       diagnostics=False):
    import torch
    from fluffy_bassoon.rl.pi05_residual import velocity_with_kv_residual
    target=batch['target32']
    noise=(torch.randn(target.shape,device=target.device,dtype=torch.float32,generator=generator)
           if fixed_noise is None else fixed_noise)
    time=(torch.as_tensor(rng.beta(1.5,1.,len(target))*.999+.001,device=target.device,dtype=torch.float32)
          if fixed_time is None else fixed_time)
    x=time[:,None,None]*noise+(1-time[:,None,None])*target
    velocity=velocity_with_kv_residual(model,SimpleNamespace(state=batch['state']),x,time,
        batch['cache'],batch['mask'],batch['features'],adapter)
    expected=noise-target
    loss=masked_mse(velocity,expected,batch['valid'])
    if not diagnostics:return loss
    # Native compute_loss uses all32 dimensions. Report the actual14 physical
    # coordinates separately, so padding fit cannot be mistaken for steering.
    details=dict(physical14_MSE=float(masked_mse(velocity[...,:14],expected[...,:14],batch['valid']).detach()),
                 padded18_MSE=float(masked_mse(velocity[...,14:],expected[...,14:],batch['valid']).detach()))
    return loss,details
