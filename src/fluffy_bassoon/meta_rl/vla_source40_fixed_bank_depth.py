"""Pinned current-bank100 ->500 state fork using the EXISTING history trainer.

No scheduling or source/teacher/robot work. Original files are read-only.
Migration changes only run/source/schedule metadata and the NEW elapsed clock;
policy (including normalizers), Adam, sampler and all RNG bytes are preserved.
Corrected validation is separate inference at100/500, never a training input or
checkpoint-selection rule. The worker must already hold its owned phase ticket.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

FORMAT = 'source40_current_bank_fixed_depth_100_500_v1'
ORIGINAL_SOURCE = '2bc15da3028ab6a3cbaed4b813f3f761a680f1d9'
MODELS = ('ad_full', 'ad_static', 'dpt_full', 'dpt_static')
CONFIG = dict(observation_dim=4104, noise_dim=32, width=128, layers=2, heads=4,
              context_length=300, initial_std=.5)
SCHEDULE = dict(updates=500, eval_every=25, validation_batches=4,
                max_elapsed_seconds=1740., checkpoint_selection='final')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or sha(path) != ref['sha256']:
        raise ValueError('Pinned current-bank/model input checksum differs')
    if ref.get('sidecar_sha256') is not None and sha(path.with_suffix('.json')) != ref['sidecar_sha256']:
        raise ValueError('Pinned derived dataset sidecar checksum differs')
    return path


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_profile(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Controlled depth profile checksum differs')
    value = json.loads(Path(path).read_text())
    if (value.get('format') != FORMAT or set(value['models']) != set(MODELS)
            or value['initial_updates'] != 100 or value['final_updates'] != 500
            or value['training_rows'] != 33729 or value['validation_rows'] != 8202
            or value['seed'] != 0 or value['config'] != CONFIG
            or value['optimization'] != dict(batch_size=8, learning_rate=3e-4, max_context=128,
                weight_decay=0., gradient_clip_norm=1., threads=4, execution_batch_size=None)
            or value['schedule'] != SCHEDULE
            or value['maximum_cumulative_GPU_seconds'] != 7200
            or value['per_model_cumulative_GPU_seconds'] != 1800
            or value['maximum_new_outputs_bytes'] != 16*1024**3
            or value['additional_robot_actions'] != 0):
        raise ValueError('Require exact four seed0 current-bank100-to500 controlled recipe')
    for name, model in value['models'].items():
        objective, mode = name.split('_')
        if (model['objective'] != objective or model['mode'] != mode
                or model['old_contract']['source_commit'] != ORIGINAL_SOURCE
                or model['old_contract']['config'] != CONFIG
                or model['old_contract']['train_sha256'] != value['data'][objective]['train']['sha256']
                or model['old_contract']['validation_sha256'] != value['data'][objective]['legacy_validation']['sha256']):
            raise ValueError('Original100 contract/data/controller differs')
    return value


def tree_sha(value):
    """Canonical existing content helper; devices/serialization order excluded."""
    import torch
    from .vla_libero import structure_digest
    def cpu(item):
        if isinstance(item, torch.Tensor): return item.detach().cpu().numpy()
        if isinstance(item, dict): return {str(k):cpu(v) for k,v in item.items()}
        if isinstance(item, (list, tuple)): return [cpu(v) for v in item]
        return item
    return structure_digest(cpu(value))


def validate_saved(saved, endpoint, specification):
    """Actual tensor/Adam inspection; call only on allocated Linux."""
    import torch
    from .vla_history import MODEL_FORMAT
    from .vla_history_train import RESUME_FORMAT
    expected = specification['old_contract']
    contract = saved.get('contract', {})
    if (saved.get('format') != RESUME_FORMAT or saved.get('update') != 100
            or any(contract.get(k) != v for k,v in expected.items())
            or endpoint.get('format') != MODEL_FORMAT
            or endpoint.get('manifest', {}).get('update') != 100
            or endpoint.get('manifest', {}).get('contract') != contract
            or endpoint.get('config') != CONFIG
            or tree_sha(saved['policy']) != tree_sha(endpoint['policy'])
            or not {'sampler','python','numpy','torch','cuda'} <= saved.get('rng', {}).keys()
            or not saved['rng']['cuda'] or not saved.get('optimizer', {}).get('state')):
        raise ValueError('Original fixed100 endpoint/full recovery/optimizer/RNG mismatch')
    allowed = {'source_commit','config','mode','objective','train_sha256','validation_sha256',
        'train_metadata','validation_metadata','seed','batch_size','max_context','learning_rate',
        'supervised_objective','status'}
    if set(contract) != allowed:
        raise ValueError('Original recipe contains unreviewed schedule/contract fields')
    normalizer = {k:saved['policy'][k] for k in ('observation_mean','observation_std')}
    if (any(v.shape != (4104,) or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in normalizer.values())
            or (normalizer['observation_std'] < 1e-4).any()):
        raise ValueError('Original shared TRAIN normalizer is missing or invalid')
    for group in saved['optimizer']['param_groups']:
        if (group['lr'] != 3e-4 or group['weight_decay'] != 0. or tuple(group['betas']) != (.9,.999)
                or group['eps'] != 1e-8 or group.get('amsgrad', False) or group.get('maximize', False)
                or group.get('foreach') is not None or group.get('fused') is not None
                or group.get('capturable',False) or group.get('differentiable',False)):
            raise ValueError('Actual original AdamW recipe differs; do not reinterpret the100 prefix')
    report = dict(update=100, policy_sha256=tree_sha(saved['policy']), optimizer_sha256=tree_sha(saved['optimizer']),
        rng_sha256=tree_sha(saved['rng']), normalizer_sha256=tree_sha(normalizer),
        train_metadata_sha256=tree_sha(contract['train_metadata']),
        legacy_validation_metadata_sha256=tree_sha(contract['validation_metadata']),
        normalizer_keys=list(normalizer), optimizer_param_groups=[{k:v for k,v in g.items() if k!='params'}
            for g in saved['optimizer']['param_groups']],
        optimizer_steps=sorted({float(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s}),
        tensor_identity_scope='original policy/Adam/RNG including exact frozen TRAIN normalizer; no reinitialization')
    return json.loads(json.dumps(report,allow_nan=False))


def inspect_original(profile, model):
    import torch
    specification = profile['models'][model]
    saved = torch.load(checked(specification['recovery']), map_location='cpu', weights_only=False)
    endpoint = torch.load(checked(specification['endpoint']), map_location='cpu', weights_only=False)
    return saved, validate_saved(saved, endpoint, specification)


def fork_state(saved, source_commit):
    """Explicit new-output metadata migration; no parameter/Adam/RNG operation."""
    result = deepcopy(saved)
    result['contract'] = dict(deepcopy(saved['contract']), source_commit=source_commit,
        max_episodes_per_window=None, schedule=dict(SCHEDULE))
    result['elapsed_seconds'] = 0.  # NEW bounded phase; historical time stays in provenance.
    result['origin'] = dict(source_commit=ORIGINAL_SOURCE, initial_updates=100,
        historical_elapsed_seconds=saved['elapsed_seconds'], metadata_fork_only=True)
    if any(tree_sha(result[k]) != tree_sha(saved[k]) for k in ('policy','optimizer','rng')):
        raise RuntimeError('State migration changed the original numerical trajectory')
    return result


def trainer_command(profile, model, output, *, source_commit, hostname):
    specification = profile['models'][model]
    data = profile['data'][specification['objective']]
    return [sys.executable, '-m', 'fluffy_bassoon.meta_rl.vla_history_train',
        '--train-data', data['train']['path'], '--train-sha256', data['train']['sha256'],
        '--validation-data', data['legacy_validation']['path'], '--validation-sha256', data['legacy_validation']['sha256'],
        '--output-dir', str(output), '--resume', str(Path(output)/'latest.pt'),
        '--mode', specification['mode'], '--seed','0','--updates','500',
        '--width','128','--layers','2','--heads','4','--context-length','300','--max-window','128',
        '--batch-size','8','--learning-rate','0.0003','--eval-every','25','--validation-batches','4',
        '--threads','4','--device','cuda:0','--max-elapsed-seconds','1740.0',
        '--checkpoint-selection','final','--source-commit',source_commit,'--expected-hostname',hostname]


def corrected_rescore(saved, data, *, mode, device):
    """Separate metric only; original model/normalizer and all training RNG saved/restored."""
    import numpy as np
    import torch
    from .vla_history import HistoryConfig, HistoryNoisePolicy, load_dataset
    from .vla_history_train import _rng_state, _restore_rng, validate
    sampler = np.random.default_rng(0)
    previous = deepcopy(_rng_state(sampler))
    saved_identity = {k:tree_sha(saved[k]) for k in ('policy','optimizer','rng')}
    try:
        trials, manifest = load_dataset(checked(data), expected_sha256=data['sha256'])
        policy = HistoryNoisePolicy(HistoryConfig(**saved['contract']['config'])).to(device)
        policy.load_state_dict(saved['policy'], strict=True)
        policy.requires_grad_(False)
        before = tree_sha(policy.state_dict())
        value = validate(policy, trials, mode=mode, batch_size=8, batches=4, max_context=128)
        if tree_sha(policy.state_dict()) != before:
            raise RuntimeError('Corrected validation modified frozen weights/normalization')
        if saved_identity != {k:tree_sha(saved[k]) for k in saved_identity}:
            raise RuntimeError('Corrected validation changed saved training state')
        return dict(update=saved['update'], metric_name=('heldout_recorded_behavior_ad'
            if saved['contract'].get('objective') == 'ad' else 'heldout_state_same_train_teacher'),
            validation_nll=value, dataset=deepcopy(data), validation_sampler_seed=734001,
            fixed_sampled_windows=True, validation_batches=4, batch_size=8, max_context=128,
            train_rng_unchanged=True, checkpoint_selection_influence=False, optimizer_updates=0,
            policy_sha256=before, normalizer_sha256=tree_sha({k:policy.state_dict()[k]
                for k in ('observation_mean','observation_std')}), transitions=manifest['transitions'])
    finally:
        _restore_rng(previous, sampler)


def clock_contract(profile, profile_sha, source_commit):
    return dict(format=FORMAT, profile_sha256=profile_sha, source_commit=source_commit,
        models=list(MODELS), maximum_cumulative_GPU_seconds=7200, per_model_cumulative_GPU_seconds=1800,
        fixed_total_updates=500, new_updates_per_model=400, source_actions=0)


def owned_attempt(root, attempt, profile, profile_sha, source_commit, model):
    for name, digest in profile['root_authority_files'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Original fixed-bank authority/grant byte binding differs')
    clock = json.loads((Path(root)/'worker-time.json').read_text())
    if clock['contract'] != clock_contract(profile, profile_sha, source_commit) or clock['worker_seconds'] != 7200:
        raise ValueError('Require the original shared7200 phase clock')
    rows = [r for r in clock['attempts'] if Path(r['path']).resolve() == Path(attempt).resolve()]
    if (len(rows) != 1 or rows[0]['status'] != 'started' or rows[0].get('lineage') != model
            or rows[0].get('lineage_seconds') != 1800 or not 0 < rows[0]['slice_seconds'] <= 1740
            or rows[0]['reserved_seconds'] != rows[0]['slice_seconds']+60
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in clock['attempts']) > 7200
            or any(sum(r.get('charged_seconds',r['reserved_seconds']) for r in clock['attempts']
                if r.get('lineage') == key) > 1800 for key in MODELS)):
        raise ValueError('Require this model\'s actual owned original1800 subcap reservation')
    return rows[0]


def verify_data(profile, model, saved):
    objective = profile['models'][model]['objective']
    data = profile['data'][objective]
    for key in ('train','derived_train','legacy_validation','corrected_validation'):
        checked(data[key])
    for field, key in (('train_metadata','train'), ('validation_metadata','legacy_validation')):
        metadata = json.loads(Path(data[key]['path']).with_suffix('.json').read_text())['metadata']
        if metadata != saved['contract'][field]:
            raise ValueError('Original training/legacy-VAL metadata differs from the exact100 contract')
    if data['train']['sha256'] != data['derived_train']['sha256']:
        raise ValueError('Derived TRAIN differs; this is no longer the fixed current bank')
    assembly = json.loads(checked(profile['derived_assembly']).read_text())
    from .vla_source40_schedule import content_hash
    if (assembly['assembly_sha256'] != profile['derived_assembly_content_sha256']
            or content_hash({k:v for k,v in assembly.items() if k!='assembly_sha256'}) != assembly['assembly_sha256']
            or assembly['datasets']['train'][objective]['sha256'] != data['train']['sha256']
            or assembly['datasets']['validation'][objective]['sha256'] != data['corrected_validation']['sha256']):
        raise ValueError('Same-TRAIN derived assembly/input binding differs')


def _guard(source_commit, hostname):
    if sys.platform != 'linux' or socket.gethostname() != hostname:
        raise ValueError('Actual checkpoint/model operations require the identified allocated Linux host')
    code = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip() != source_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()):
        raise ValueError('Require exact clean reviewed current code')


def runtime_code():
    root=Path(__file__).resolve().parent
    names=('vla_source40_fixed_bank_depth.py','vla_history.py','vla_history_train.py','gtrxl.py',
        'vla_libero.py','warmstart.py','vla_source40_subset_training.py','vla_source40_ppo_worker.py')
    return {name:sha(root/name) for name in names}


def runtime(profile, *, gpu):
    import torch
    import numpy as np
    if torch.__version__ != profile['runtime']['torch'] or torch.version.cuda != profile['runtime']['cuda']:
        raise ValueError(f"Original qualified runtime required; observed Torch{torch.__version__}/CUDA{torch.version.cuda}")
    value = dict(torch=str(torch.__version__), cuda=torch.version.cuda, executable=sys.executable,
        python=sys.version,numpy=np.__version__,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)
    if gpu:
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != profile['runtime']['device']:
            raise ValueError('Require one actually allocated GPU of the original qualified kind')
        value.update(device=torch.cuda.get_device_name(0),
            total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
            allocated_device_bytes=torch.cuda.memory_allocated(0))
    return value


def gpu_restore_check(saved):
    """Actual same-ticket restore check; zero updates and no VLA/teacher calls."""
    import numpy as np
    import torch
    from .vla_history import HistoryConfig,HistoryNoisePolicy
    from .vla_history_train import _restore_rng,_rng_state
    policy=HistoryNoisePolicy(HistoryConfig(**saved['contract']['config'])).to('cuda:0')
    optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
    policy.load_state_dict(saved['policy'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    sampler=np.random.default_rng(0)
    _restore_rng(saved['rng'],sampler)
    result=dict(policy_sha256=tree_sha(policy.state_dict()),optimizer_sha256=tree_sha(optimizer.state_dict()),
        rng_sha256=tree_sha(_rng_state(sampler)),optimizer_updates=0,
        allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))
    if any(result[key+'_sha256']!=tree_sha(saved[key]) for key in ('policy','optimizer','rng')):
        raise ValueError('Actual CUDA model/Adam/private-global RNG restore differs; no numerical repair')
    return result


def output_guard(root, profile):
    size = 0
    for path in Path(root).rglob('*'):
        try:
            if path.is_file(): size += path.stat().st_size
        except FileNotFoundError: pass  # Another model's atomic temporary just moved.
    if size+128*1024**2 > profile['maximum_new_outputs_bytes']:
        raise RuntimeError('Shared new-output cap cannot reserve another full recovery/export')
    return size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('inspect','train'))
    for name in ('profile','output'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('profile-sha256','source-commit','expected-hostname'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--phase-root', type=Path)
    parser.add_argument('--attempt', type=Path)
    parser.add_argument('--inspection', type=Path)
    parser.add_argument('--inspection-sha256')
    parser.add_argument('--resume-phase',action='store_true')
    args = parser.parse_args(argv)
    _guard(args.source_commit, args.expected_hostname)
    profile = load_profile(args.profile, args.profile_sha256)
    if args.operation == 'inspect':
        started = time.monotonic()
        actual_runtime = runtime(profile,gpu=False)
        import torch
        torch.set_num_threads(1)  # Bounded read-only CPU tensor inspection, not training.
        args.output.mkdir(parents=True, exist_ok=False)
        results = {};model=None
        receipt=dict(format=FORMAT,actual_original_state_inspection=False,completed=False,
            models=results, profile_sha256=args.profile_sha256, source_commit=args.source_commit,
            model_construction=False, forward_inference=False, optimizer_updates=0,
            runtime=actual_runtime,runtime_code_sha256=runtime_code(),inspection_threads=1,
            cuda_restore_verified=False,cuda_restore_scope='Same-model GPU startup via existing trainer full RNG restore')
        try:
            for model in MODELS:
                if time.monotonic()-started > 45:
                    raise RuntimeError('Bounded actual saved-state inspection time exhausted')
                saved,report=inspect_original(profile,model)
                verify_data(profile,model,saved)
                results[model]=report
                write(args.output/'inspection.json',receipt)
            if len({r['normalizer_sha256'] for r in results.values()})!=1:
                raise ValueError('Actual100 normalizers differ across four matched controllers')
            receipt.update(actual_original_state_inspection=True,completed=True)
        except Exception as error:
            receipt.update(failed_model=model,error=repr(error))
            raise
        finally:
            receipt.update(elapsed_seconds=time.monotonic()-started,
                maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            write(args.output/'inspection.json',receipt)
        return
    if (args.model is None or args.phase_root is None or args.attempt is None
            or args.inspection is None or args.inspection_sha256 is None):
        raise ValueError('Training requires one model and its already owned phase attempt')
    if args.output.resolve() != (args.phase_root/'models'/args.model).resolve():
        raise ValueError('New model outputs must stay inside their original controlled phase')
    inspection = json.loads(checked(dict(path=str(args.inspection),sha256=args.inspection_sha256)).read_text())
    if (inspection['profile_sha256'] != args.profile_sha256
            or inspection.get('actual_original_state_inspection') is not True or inspection.get('completed') is not True
            or inspection['runtime_code_sha256'] != runtime_code()):
        raise ValueError('Actual prelaunch original-state inspection binding differs')
    row = owned_attempt(args.phase_root,args.attempt,profile,args.profile_sha256,args.source_commit,args.model)
    deadline = row['started_unix']+row['slice_seconds']-30
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION','baige'),os.environ)
    import torch
    actual_runtime = runtime(profile,gpu=True)
    saved, report = inspect_original(profile,args.model)
    if report != inspection['models'][args.model]:
        raise ValueError('Actual GPU-startup original model/Adam/normalizer/RNG differs from prelaunch inspection')
    verify_data(profile,args.model,saved)
    args.output.mkdir(parents=True,exist_ok=args.resume_phase)
    output_guard(args.phase_root,profile)
    write(args.output/'runtime.json',actual_runtime)
    write(args.output/'gpu-restore-check.json',gpu_restore_check(saved))
    if not args.resume_phase: write(args.output/'original-state-inspection.json',report)
    specification = profile['models'][args.model]
    corrected = profile['data'][specification['objective']]['corrected_validation']
    if not args.resume_phase:
        write(args.output/'corrected-validation-100.json',corrected_rescore(saved,corrected,mode=specification['mode'],device='cuda:0'))
        migrated = fork_state(saved,args.source_commit)
        from .warmstart import atomic_save
        atomic_save(args.output/'latest.pt',migrated)
        write(args.output/'contract.json',migrated['contract'])
        write(args.output/'migration.json',dict(format=FORMAT,original=specification,
            original_identity=report,profile_sha256=args.profile_sha256,new_recovery_sha256=sha(args.output/'latest.pt'),
            metadata_changes=['source_commit','schedule','max_episodes_per_window','new elapsed clock'],
            training_data_changed=False,numerical_state_changed=False,initial_updates=100,new_requested_updates=400))
    else:
        migration = json.loads((args.output/'migration.json').read_text())
        if migration['profile_sha256'] != args.profile_sha256 or migration['original_identity'] != report:
            raise ValueError('Manual continuation changed its original numerical prefix or controlled phase')
    latest = torch.load(args.output/'latest.pt',map_location='cpu',weights_only=False)
    if (latest['contract'] != fork_state(saved,args.source_commit)['contract']
            or not 100 <= latest['update'] <= 500):
        raise ValueError('Owned continuation checkpoint differs from original fixed500 trajectory')
    command = trainer_command(profile,args.model,args.output,source_commit=args.source_commit,hostname=args.expected_hostname)
    remaining = deadline-time.time()
    if remaining <= 60: raise RuntimeError('Owned inclusive slice has insufficient time after restore/rescore')
    if latest['update'] < 500: subprocess.run(command,check=True,timeout=remaining)
    result = torch.load(args.output/'latest.pt',map_location='cpu',weights_only=False)
    if result['update'] != 500: raise ValueError('Fixed500 endpoint incomplete; retain full recovery, no endpoint claim')
    if tree_sha({k:result['policy'][k] for k in ('observation_mean','observation_std')}) != report['normalizer_sha256']:
        raise RuntimeError('Frozen original TRAIN normalizer changed')
    write(args.output/'corrected-validation-500.json',corrected_rescore(result,corrected,mode=specification['mode'],device='cuda:0'))
    output_guard(args.phase_root,profile)
    from .vla_source40_subset_training import immutable_copy
    final_recovery_sha=immutable_copy(args.output/'latest.pt',args.output/'endpoint-500-recovery.pt')
    final_policy_sha=immutable_copy(args.output/'candidate.pt',args.output/'endpoint-500.pt')
    write(args.output/'result.json',dict(format=FORMAT,completed=True,model=args.model,
        initial_updates=100,final_updates=500,new_optimizer_updates=400,
        recovery_sha256=final_recovery_sha,candidate_sha256=final_policy_sha,
        recovery_path=str(args.output/'endpoint-500-recovery.pt'),candidate_path=str(args.output/'endpoint-500.pt'),
        original_identity=report,profile_sha256=args.profile_sha256,source_commit=args.source_commit,
        inspection_source_commit=inspection['source_commit'],runtime_code_sha256=runtime_code(),
        source_actions=0,interpretation='Fixed current-bank optimization-depth control; no robot/memory/teacher competence claim'))


if __name__ == '__main__':
    main()
