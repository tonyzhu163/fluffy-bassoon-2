"""Bounded source-subset optimizer-scale qualification, with immutable milestones.

Four matched AD/DPT full/static controllers each follow one optimizer trajectory.
25/100/500 are update checkpoints, not additional source interactions or three
independent training runs. Full40 production admission is unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from .vla_history import file_sha256
from .vla_source40_training import assemble, load_assembly, load_source, OBJECTIVES, MODES


FORMAT = 'openpi_pi05_source40_subset_scales_v1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_scales(values):
    if (not values or any(type(value) is not int or not 1 <= value <= 500 for value in values)
            or values != sorted(set(values))):
        raise ValueError('Require increasing distinct positive optimizer milestones, capped at500')
    return values


def training_command(args, assembly, *, objective, mode, updates, resume):
    """Every milestone continues the same controller directory and optimizer."""
    train, validation = (assembly['datasets'][partition][objective] for partition in ('train', 'validation'))
    output = args.output / 'students' / f'{objective}-{mode}-seed-{args.seed}'
    dataset_root = args.assembly.parent if args.assembly is not None else args.output / 'dataset'
    command = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.vla_history_train',
        '--train-data', str(dataset_root / train['path']), '--train-sha256', train['sha256'],
        '--validation-data', str(dataset_root / validation['path']), '--validation-sha256', validation['sha256'],
        '--output-dir', str(output), '--mode', mode, '--seed', str(args.seed), '--updates', str(updates),
        '--source-commit', args.source_commit, '--expected-hostname', args.expected_hostname,
        '--max-elapsed-seconds', str(args.max_controller_seconds)]
    for name in ('batch_size', 'learning_rate', 'width', 'layers', 'heads', 'context_length',
                 'max_window', 'eval_every', 'validation_batches', 'threads', 'device'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if resume:
        command += ['--resume', str(output / 'latest.pt')]
    return command


def immutable_copy(source, destination):
    source, destination = Path(source), Path(destination)
    digest = file_sha256(source)
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError('Existing immutable milestone differs from its full recovery')
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    if temporary.exists():
        temporary.rename(temporary.with_name(temporary.name + f'-uncommitted-{time.time_ns()}'))
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != digest:
        raise RuntimeError('Milestone recovery copy checksum differs')
    temporary.replace(destination)
    return digest


def archive_scale(recovery, output, *, updates, assembly_sha256, exposure):
    import torch
    from .vla_history import HistoryConfig, HistoryNoisePolicy, export_policy, load_policy
    from .vla_history_train import RESUME_FORMAT
    recovery, output = Path(recovery), Path(output)
    saved = torch.load(recovery, map_location='cpu', weights_only=False)
    if (saved.get('format') != RESUME_FORMAT or saved.get('update') != updates
            or not saved.get('optimizer') or not {'sampler', 'python', 'numpy', 'torch', 'cuda'} <= set(saved.get('rng', {}))):
        raise ValueError('Milestone requires the exact update and complete optimizer/sampler/RNG recovery')
    output.mkdir(parents=True, exist_ok=True)
    recovery_path = output / 'recovery.pt'
    recovery_sha = immutable_copy(recovery, recovery_path)
    policy = HistoryNoisePolicy(HistoryConfig(**saved['contract']['config']))
    policy.load_state_dict(saved['policy'], strict=True)
    manifest = dict(contract=saved['contract'], update=updates, selection='fixed requested final update',
        assembly_sha256=assembly_sha256, full_recovery_sha256=recovery_sha,
        qualification_subset=True, source_exposure=exposure,
        status='Unvalidated subset robot-control candidate; optimizer scale is not source-data scale')
    final_path = output / 'final.pt'
    if final_path.exists():
        previous, previous_manifest = load_policy(final_path, expected_sha256=file_sha256(final_path))
        if previous_manifest != manifest or any(not torch.equal(value, previous.state_dict()[key])
                for key, value in policy.state_dict().items()):
            raise ValueError('Existing immutable milestone policy differs')
    else:
        export_policy(final_path, policy, manifest=manifest)
    return dict(objective=saved['contract']['objective'], mode=saved['contract']['mode'],
        seed=saved['contract']['seed'], updates=updates, elapsed_seconds=saved['elapsed_seconds'],
        final_path=str(final_path), final_sha256=file_sha256(final_path),
        full_recovery_path=str(recovery_path), full_recovery_sha256=recovery_sha)


def frozen_baseline(index_path):
    index_path = Path(index_path).resolve()
    source = load_source(index_path, 'train', allow_small_pilot=True)
    selected = source['selected']
    contract_path = index_path.parent / 'contract.json'
    return dict(task_key=source['task'], index_path=str(index_path), index_sha256=source['index_sha256'],
        checkpoint_path=str(index_path.parent / selected['source_checkpoint']),
        checkpoint_sha256=selected['source_checkpoint_sha256'], contract_path=str(contract_path),
        contract_sha256=file_sha256(contract_path), source_updates=source['costs']['source_updates'],
        source_generation_cost=source['costs'], executed_history_chunks=selected['transitions'],
        interface='Official pixel-SAC image64/proprio8 actor; students use frozen VLA-prefix4104. Same frozen motor and latent-noise map.',
        status='Frozen source actor candidate; no independent expert or improvement claim')


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qualification-subset', action='store_true', required=True)
    ap.add_argument('--train-index', type=Path, nargs='+', required=True)
    ap.add_argument('--validation-index', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--assembly', type=Path, help='Optional immutable subset-data assembly; original source indices/costs remain required')
    ap.add_argument('--assembly-file-sha256')
    ap.add_argument('--scales', type=int, nargs='+', default=[25, 100, 500])
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--expected-hostname', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-controller-seconds', type=float, default=300.)
    ap.add_argument('--max-total-seconds', type=float, default=1500.)
    for name, default in (('batch-size', 8), ('width', 128), ('layers', 2), ('heads', 4),
                          ('context-length', 300), ('max-window', 128), ('eval-every', 25),
                          ('validation-batches', 4), ('threads', 4)):
        ap.add_argument('--' + name, type=int, default=default)
    ap.add_argument('--learning-rate', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    validate_scales(args.scales)
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or not 0 < args.max_controller_seconds <= 300 or not 0 < args.max_total_seconds <= 1800):
        raise ValueError('Require allocated Linux and bounded per-controller/total qualification time')
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source commit')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    args.output = args.output.resolve()
    if bool(args.assembly) != bool(args.assembly_file_sha256):
        raise ValueError('A supplied subset assembly needs its explicit file checksum')
    supplied = None
    if args.assembly is not None:
        args.assembly = args.assembly.resolve()
        if file_sha256(args.assembly) != args.assembly_file_sha256:
            raise ValueError('Supplied subset assembly checksum differs')
        supplied = load_assembly(args.assembly, allow_small_pilot=True)
        if supplied.get('small_pilot') is not True:
            raise ValueError('This bounded route accepts only an explicitly marked subset assembly')
        for partition, paths in (('train', args.train_index), ('validation', args.validation_index)):
            expected = {(str(path.resolve()), file_sha256(path)) for path in paths}
            if {(row['index_path'], row['index_sha256']) for row in supplied['selected_sources'][partition]} != expected:
                raise ValueError('Subset assembly changed the original source exposure/cost bank')
    assembly_path = args.assembly if args.assembly is not None else args.output / 'dataset/assembly.json'
    baseline = frozen_baseline(args.train_index[0])
    contract = dict(format=FORMAT, qualification_subset=True, source_commit=args.source_commit,
        train_indices=[dict(path=str(path.resolve()), sha256=file_sha256(path)) for path in args.train_index],
        validation_indices=[dict(path=str(path.resolve()), sha256=file_sha256(path)) for path in args.validation_index],
        scales=args.scales, seed=args.seed, frozen_source_sac=baseline,
        provided_assembly=(dict(path=str(args.assembly), file_sha256=args.assembly_file_sha256,
            assembly_sha256=supplied['assembly_sha256']) if supplied is not None else None),
        optimization={name: getattr(args, name) for name in ('batch_size', 'learning_rate', 'width', 'layers', 'heads',
            'context_length', 'max_window', 'eval_every', 'validation_batches', 'threads', 'device')},
        max_controller_seconds=args.max_controller_seconds, max_total_seconds=args.max_total_seconds,
        chronology='One uninterrupted optimizer trajectory per controller, resumed at milestones;500 final updates rather than25+100+500',
        interpretation='Source subset development only; no full40 coverage, expert-label, new-task or adaptation claim')
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('Subset scale resume contract differs')
        assembly = supplied if supplied is not None else load_assembly(assembly_path, allow_small_pilot=True)
        old_progress = json.loads((args.output / 'progress.json').read_text())
        elapsed_before = old_progress['elapsed_seconds']
    else:
        write_json(args.output / 'contract.json', contract)
        assembly = supplied if supplied is not None else assemble(
            args.train_index, args.validation_index, args.output / 'dataset', allow_small_pilot=True)
        elapsed_before = 0.
        write_json(args.output / 'progress.json', dict(elapsed_seconds=0., milestones=[]))
    exposure = dict(training_task_keys=assembly['source_task_keys'],
        validation_task_keys=assembly.get('validation_task_keys', assembly['source_task_keys']),
        selected_source_histories=assembly['selected_sources'], small_pilot=True,
        validation='Source-local cases already used for NLL; subsequent robot probes on those cases are development evidence')
    started, archived = time.monotonic(), {}
    import torch
    for objective in OBJECTIVES:
        for mode in MODES:
            live = args.output / 'students' / f'{objective}-{mode}-seed-{args.seed}' / 'latest.pt'
            for scale in args.scales:
                archive = args.output / 'scales' / f'update-{scale:06d}' / f'{objective}-{mode}-seed-{args.seed}'
                if (archive / 'recovery.pt').exists():
                    recovery = archive / 'recovery.pt'
                else:
                    update = torch.load(live, map_location='cpu', weights_only=False)['update'] if live.exists() else 0
                    if update > scale:
                        raise ValueError('Missing immutable earlier milestone; do not relabel later weights')
                    remaining = args.max_total_seconds - elapsed_before - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError('Subset qualification reached its total time ceiling; full recoveries preserved')
                    if update < scale:
                        command = training_command(args, assembly, objective=objective, mode=mode,
                                                   updates=scale, resume=live.exists())
                        try:
                            subprocess.run(command, check=True, timeout=remaining)
                        finally:
                            write_json(args.output / 'progress.json', dict(
                                elapsed_seconds=elapsed_before + time.monotonic() - started,
                                milestones=list(archived.values()),
                                source_collection_costs_count_once=assembly['collection_costs_count_once']))
                    recovery = live
                row = archive_scale(recovery, archive, updates=scale,
                    assembly_sha256=assembly['assembly_sha256'], exposure=exposure)
                archived[(objective, mode, scale)] = row
                scale_rows = [archived[(o, m, scale)] for o in OBJECTIVES for m in MODES if (o, m, scale) in archived]
                matrix = dict(contract=dict(format='openpi_pi05_libero40_matched_training_v1',
                    assembly_sha256=assembly['assembly_sha256'], source_commit=args.source_commit,
                    seeds=[args.seed], qualification_subset=True), completed=len(scale_rows) == 4,
                    controllers=scale_rows, frozen_source_sac=baseline)
                write_json(archive.parent / 'index.json', matrix)
                write_json(args.output / 'progress.json', dict(elapsed_seconds=elapsed_before + time.monotonic() - started,
                    milestones=list(archived.values()), source_collection_costs_count_once=assembly['collection_costs_count_once']))
    scales = [dict(updates=scale, index_path=str(args.output / 'scales' / f'update-{scale:06d}' / 'index.json'),
        index_sha256=file_sha256(args.output / 'scales' / f'update-{scale:06d}' / 'index.json')) for scale in args.scales]
    result = dict(format=FORMAT, completed=True, qualification_subset=True, seed=args.seed, scales=scales,
        assembly_path=str(assembly_path), assembly_sha256=assembly['assembly_sha256'],
        assembly_file_sha256=file_sha256(assembly_path), frozen_source_sac=baseline,
        source_collection_costs_count_once=assembly['collection_costs_count_once'],
        unique_controller_optimizer_updates=4 * args.scales[-1], elapsed_seconds=elapsed_before + time.monotonic() - started,
        source_exposure=exposure, interpretation=contract['interpretation'])
    write_json(args.output / 'scales.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
