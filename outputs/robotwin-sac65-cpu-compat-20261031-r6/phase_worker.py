"""Owned CPU phase worker; import firewall precedes third-party site paths."""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

SHARED_RUNTIME_METADATA_SHA='715a0ab60975cfa63a3d19ce4bcd5cd58c355db799aa1b6884f4027765c49322'

BANNED = ('jax', 'jaxlib', 'openpi', 'sapien', 'rlinf', 'mujoco', 'mani_skill',
          'fluffy_bassoon.rl.robotwin_noise', 'fluffy_bassoon.rl.robotwin_assets',
          'fluffy_bassoon.rl.robotwin_production', 'fluffy_bassoon.rl.metaworld_pi05',
          'fluffy_bassoon.rl.maniskill_pi05_noise', 'fluffy_bassoon.rl.robotwin_campaign_budget',
          'fluffy_bassoon.meta_rl.robotwin_trial_resume', 'fluffy_bassoon.vla', 'lerobot.policies')


def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.')
               or (prefix.endswith(('robotwin_production', 'metaworld_pi05')) and name.startswith(prefix))
               for prefix in BANNED)


class CPUImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.denied = []

    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            self.denied.append(fullname)
            raise ImportError('CPU compatibility gate forbids simulator/frozen-VLA import: ' + fullname)
        return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def approved_shared_site(ctx):
    ref=ctx['shared_runtime_metadata'];path=Path(ref['path'])
    if (ref['sha256']!=SHARED_RUNTIME_METADATA_SHA or path.is_symlink()
            or not path.is_file() or sha(path)!=SHARED_RUNTIME_METADATA_SHA):
        raise ValueError('Approved shared-runtime metadata changed')
    metadata=json.loads(path.read_text())
    own=Path(ctx['site_packages']);pth=Path(metadata['pth_path'])
    if pth!=own/'shared_runtime.pth' or pth.is_symlink() or not pth.is_file():
        raise ValueError('Expected the existing venv literal shared_runtime.pth')
    raw=pth.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=metadata['pth_sha256']:
        raise ValueError('Shared-runtime literal-path bytes changed')
    lines=[line.strip() for line in raw.decode('utf-8').splitlines() if line.strip()]
    if lines!=[metadata['literal_path']] or not Path(lines[0]).is_absolute():
        raise ValueError('Only the single approved literal path is allowed; no pth execution')
    shared=Path(lines[0]);torch_init=shared/'torch/__init__.py'
    if (not shared.is_dir() or not torch_init.is_file()
            or sha(torch_init)!=metadata['torch_init_sha256']
            or sorted(p.name for p in shared.glob('torch-*.dist-info'))!=metadata['torch_dist_info']):
        raise ValueError('Approved existing shared Torch package metadata differs')
    return shared,dict(metadata=ref,pth_path=str(pth),pth_sha256=metadata['pth_sha256'],
        shared_site_packages=str(shared),torch_init=str(torch_init),torch_init_sha256=metadata['torch_init_sha256'],
        torch_dist_info=metadata['torch_dist_info'],pth_processing_performed=False)


def verify_source65_index(index, ctx, projection_path):
    expected = dict(complete_K4_trials=390, episodes=1560,
                    committed_source_actions=651638, selected_actions=651638)
    snapshot = ctx['tip']['checkpoint']
    iterations = index.get('source_contract', {}).get('source_iterations')
    if (index.get('partition') != 'train' or index.get('source_method') != 'sac'
            or index.get('teacher_method') != 'sac' or set(index.get('datasets', {})) != {'ad', 'dpt'}
            or any(index.get('counts', {}).get(k) != v for k, v in expected.items())
            or index.get('source_recovery_sha256') != snapshot['sha256']
            or index.get('teacher_recovery_sha256') != snapshot['sha256']
            or not isinstance(iterations, dict) or iterations.get('count') != 100
            or iterations.get('trials_per_iteration') != 6 or iterations.get('episodes_per_trial') != 4):
        raise ValueError('Wrong closed source65 index/schema/source/teacher/counters')
    inputs = index['input_sha256']
    if (inputs.get(str(Path(snapshot['path']).resolve())) != snapshot['sha256']
            or inputs.get(str(Path(projection_path).resolve())) != sha(projection_path)):
        raise ValueError('Index is not linked to this closed source65 projection/checkpoint')


def finalbank_phase(phase, ctx, code, receipts, allocation):
    """Owner operational profile around the existing7401 APIs, never a converter."""
    f=ctx['finalbank'];base=Path(ctx['finalbank_root']);private=base/'private'
    residual=base.name in ('robotwin-final100-ad-export-20261031-r2','robotwin-final100-ad-export-20261031-r3')
    projection=Path(f['reused_projection']['path']).parent if residual else private/'projection'
    exports=private/'exports'
    if (base not in (Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-final100-ad-export-20261031-r1'),
                     Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-final100-ad-export-20261031-r2'),
                     Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-final100-ad-export-20261031-r3'))
            or (ctx['code_commit']!=ctx['compatibility_source_pin'] if residual else
                ctx['code_commit']!='7401cd50c038a4cb35149eceeb72128e9a368cc3')
            or f['ancestry']['sha256']!='0af5082349ec1a032eff6b752e3b91fd473b8c922a394c496b4f75bceff68c7e'
            or f['tip']['sha256']!='5b978a88c423ccdc7e1921f309ec047d0fc6cfe034d07e1f9e5523e6cc2f25d4'
            or f['source_stage']['sha256']!='59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90'
            or f['final_checkpoint']['sha256']!='f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8'):
        raise ValueError('Not the prepared closed final62/source100 AD profile')
    def read_ref(ref):
        if Path(ref['path']).is_symlink() or sha(ref['path'])!=ref['sha256']:
            raise ValueError('Finalbank input changed: '+ref['path'])
        return json.loads(Path(ref['path']).read_text())
    ancestry=read_ref(f['ancestry']);tip=read_ref(f['tip'])
    if (ancestry['root_count']!=62 or ancestry['tip_manifest']!=f['tip']
            or tip['checkpoint']!=f['final_checkpoint']
            or tip['expected']['source_steps']!=1003790
            or tip['expected']['completed_source_trials']!=600
            or tip['expected']['completed_source_iterations']!=100):
        raise ValueError('Wrong actual final validation-inclusive ancestry/tip')
    tools=code/'outputs/robotwin-finalbank-offline-plan-20261031-r1'
    if phase=='reuse':
        free=shutil.disk_usage(base).free
        if free<16<<30:raise ValueError('Need16GiB free before readonly projection reuse/export')
        probe=private/'reuse-write-probe'
        with probe.open('xb') as stream:
            stream.write(b'final100-ad-reuse');stream.flush();os.fsync(stream.fileno())
        probe.unlink()
        if (not residual or projection!=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-final100-ad-export-20261031-r1/private/projection')
                or f['reused_projection']['sha256']!='fd0a3302d0ff72107e1f690ef5349825cf927da12d10da6ac457959dcf72b4cf'
                or (projection/'ASSEMBLY-INCOMPLETE').exists()):
            raise ValueError('Only the complete paid r1 projection may be reused readonly')
        projected=read_ref(f['reused_projection']);terminal=read_ref(f['producer_terminal']);cleanup=read_ref(f['producer_cleanup'])
        retained=read_ref(f['producer_storage'])
        phase_proof=[x for x in terminal['phases'] if x['name']=='finalbank-projection']
        if (terminal['run_id']!='robotwin-final100-ad-export-20261031-r1' or terminal['status']!='incomplete'
                or terminal['cleanup_error'] is not None or len(phase_proof)!=1
                or phase_proof[0]['returncode']!=0 or not phase_proof[0]['cleanup']['clean']
                or not cleanup['clean'] or cleanup['survivors'] or not cleanup['adopted']['clean']
                or retained['logical_bytes']>89<<30
                or projected['status']!='byte_projection_only_not_export_admission'
                or projected['purpose']!='finalbank' or projected['source_rounds']!=100
                or projected['ancestry']!=f['ancestry'] or projected['tip_manifest']!=f['tip']
                or projected['source_checkpoint']!=f['final_checkpoint']
                or projected['unique_files']!=218673 or projected['unique_file_bytes']!=87636354374):
            raise ValueError('Paid projection or producer closure differs')
        if sha(ctx['chain_reader'])!='67db52fa8f6fa4e8bb56bc6e5cea9f531b5fe41716845350f3372226c5de77a0':
            raise ValueError('Qualified ancestry reader changed')
        chain=load('unchanged_c62_projection_reuse',ctx['chain_reader'])
        roots=[str(Path(x).resolve()) for x in chain.corpus_chain(f['tip'])]
        if roots!=ancestry['ordered_corpus_roots'] or roots!=[x['root'] for x in projected['roots']]:
            raise ValueError('Reused projection ordered lineage changed')
        (base/'projection-reuse.json').write_text(json.dumps(dict(status='readonly_complete_projection_reuse',
            projection=f['reused_projection'],producer_terminal=f['producer_terminal'],producer_cleanup=f['producer_cleanup'],
            roots=62,unique_files=218673,unique_file_bytes=87636354374,retained_logical_storage_reserve_bytes=89<<30,
            free_bytes=free,write_fsync_probe=True,
            raw_rehash_repeated=False,original_producer_fullSHA_phase_pass=True,
            source_case_and_array_admission='Still required in unchanged export/load validators'),indent=2)+'\n')
    elif phase=='inventory':
        count=total=0;device=base.stat().st_dev;roots=[]
        free=shutil.disk_usage(base).free
        if free<16<<30:raise ValueError('Need16GiB free and approved128GiB new logical storage')
        probe=private/'write-probe'
        with probe.open('xb') as stream:
            stream.write(b'final100-ad');stream.flush();os.fsync(stream.fileno())
        probe.unlink()
        for name in ancestry['ordered_corpus_roots']:
            root=Path(name)
            if root.resolve()!=root or root.is_symlink():raise ValueError('Noncanonical final source root')
            n=size=0
            for folder in ('episodes','frames','policies'):
                top=root/folder
                if not top.is_dir() or top.is_symlink():raise ValueError('Missing/redirected corpus folder')
                for path in top.rglob('*'):
                    if time.monotonic()>=ctx['phase_deadline']:raise TimeoutError('Final inventory deadline')
                    if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Source path escape')
                    if path.is_dir():continue
                    st=path.stat()
                    if not path.is_file() or st.st_dev!=device:raise ValueError('Same-filesystem hardlinks required')
                    n+=1;size+=st.st_size;count+=1;total+=st.st_size
                    if count>500000 or total>88<<30:raise ValueError('Final storage/I/O eligibility exceeded')
            roots.append(dict(root=name,files=n,bytes=size))
        value=dict(roots=roots,files=count,bytes=total,hardlink_only=True,copy_fallback=False,
            filesystem_device=device,free_bytes=free,max_source_bytes=88<<30,
            max_new_logical_bytes=128<<30,export_metadata_headroom_bytes=40<<30,
            conservative_read_estimate_bytes=8*total+64*(1<<30),
            read_estimate_ceiling_bytes=768<<30,read_estimate_is_not_syscall_measurement=True)
        (base/'io-inventory.json').write_text(json.dumps(value,indent=2)+'\n')
    elif phase=='projection':
        admission=dict(assembly_cpu_gate_passed=True,closed_source100_authorized=True,method='sac',
            lineage_commit='db0184089fa88a5f96c153f9b80029bb91b77def',ancestry=f['ancestry'])
        path=private/'projection-admission.json';path.write_text(json.dumps(admission,indent=2)+'\n')
        project=load('existing_final100_projector',tools/'project_corpus.py')
        sys.argv=[str(tools/'project_corpus.py'),'--purpose','finalbank','--admission',str(path),
            '--admission-sha256',sha(path),'--chain-reader',ctx['chain_reader'],
            '--output',str(projection),'--expected-hostname',allocation['resource']['hostname']]
        project.main()
    elif phase in ('train','validation'):
        from fluffy_bassoon.meta_rl.robotwin_history_corpus import main as export_main
        args=['--corpus-root',str(projection),'--source-recovery',f['source_stage']['path'],
            '--source-sha256',f['source_stage']['sha256'],'--source-method','sac',
            '--objectives','ad' if phase=='train' else 'behavior_cloning','--partition',phase,
            '--output',str(exports/phase),'--device','cpu','--threads','4',
            '--source-commit',ctx['code_commit'],'--expected-hostname',allocation['resource']['hostname']]
        if phase=='validation':args+=['--validation-iteration','100']
        export_main(args)
    elif phase=='admission':
        runner=load('existing_final_index_guard',tools/'run_offline.py')
        from fluffy_bassoon.meta_rl.robotwin_history import load_dataset,validate_partitions
        bindings=dict(projection_root=str(projection),exports_root=str(exports),source100_stage=f['source_stage'])
        for partition in ('train','validation'):
            path=exports/partition/'index.json'
            bindings[partition+'_index']=dict(path=str(path),sha256=sha(path))
        projected=json.loads((projection/'projection.json').read_text())
        runner.verify_final_indexes(bindings,projected,objectives=('ad',))
        metadata={};crcs={}
        for partition,objective,trials,episodes,actions in (
                ('train','ad',600,2400,1003790),('validation','behavior_cloning',6,24,10180)):
            ref=bindings[partition+'_index']
            sequences,m=load_dataset(ref['path'],ref['sha256'],objective)
            if (m['counts']['complete_K4_trials']!=trials or m['counts']['episodes']!=episodes
                    or m['counts']['selected_actions']!=actions
                    or sum(int(s['durations'].sum()) for s in sequences)!=actions):
                raise ValueError('Final source/validation actual coverage differs')
            with zipfile.ZipFile(m['datasets'][objective]['path']) as archive:
                if archive.testzip() is not None:raise ValueError('Final dataset CRC failure')
                crcs[partition]=len(archive.infolist())
            metadata[partition]=m
            del sequences
        validate_partitions(metadata['train'],metadata['validation'])
        value=dict(status='final100_AD_BC_CPU_payload_admitted',bindings=bindings,
            projection_sha256=sha(projection/'projection.json'),counts={k:v['counts'] for k,v in metadata.items()},
            datasets={k:v['datasets'] for k,v in metadata.items()},ZIP_members_CRC_pass=crcs,
            source_actor_counter_binding='existing7401 CorpusReader, not a new teacher gate',
            teacher_used=False,additional_robot_actions=0,VLA_queries=0,SGD=0,
            model_payloads=None,fit_or_readout_admission=False)
        (base/'payload-admission.json').write_text(json.dumps(value,indent=2)+'\n')
    # Count linked payloads conservatively as new logical bytes, even though
    # blobs share the immutable originals' inodes. Never remove partial output.
    size=0
    for path in base.rglob('*'):
        if time.monotonic()>=ctx['phase_deadline']:raise TimeoutError('Final output accounting deadline')
        if path.is_symlink():raise ValueError('Redirected final output')
        if path.is_file():size+=path.stat().st_size
        if size+(89<<30 if residual else 0)>128<<30:raise ValueError('Approved aggregate logical storage ceiling exceeded')
    (receipts/'storage.json').write_text(json.dumps(dict(logical_bytes=size,ceiling_bytes=128<<30,
        retained_previous_output_reserve_bytes=89<<30 if residual else 0,
        aggregate_conservative_logical_bytes=size+(89<<30 if residual else 0)))+'\n')


def main():
    started = time.monotonic()
    guard = CPUImportGuard(); sys.meta_path.insert(0, guard)
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError('Forbidden module was loaded before the import firewall')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('tests', 'projection', 'export', 'audit', 'software-tests', 'source-admit',
        'finalbank-inventory','finalbank-projection','finalbank-reuse','finalbank-train','finalbank-validation','finalbank-admission'), required=True)
    p.add_argument('--context', type=Path, required=True); p.add_argument('--context-sha256', required=True)
    a = p.parse_args()
    if sha(a.context) != a.context_sha256:
        raise ValueError('Gate context changed')
    ctx = json.loads(a.context.read_text()); code, out = Path(ctx['code_root']), Path(ctx['output'])
    if sha(ctx['allocation_receipt']) != ctx['allocation_receipt_sha256']:
        raise ValueError('Actual0GPU pre-mask receipt changed')
    allocation = json.loads(Path(ctx['allocation_receipt']).read_text())
    if allocation['resource']['allocated_gpus'] != 0 or not allocation['actual_zero_gpu_attested']:
        raise ValueError('Require the actual pre-mask zero-GPU receipt')
    # -S avoids sitecustomize/.pth execution. Add only existing library paths,
    # then the exact candidate source/tests. No RLinf/OpenPI/native source path.
    site = Path(ctx['site_packages'])
    if not site.is_dir():
        raise ValueError('Existing environment site-packages path is missing')
    shared,library_proof=approved_shared_site(ctx)
    sys.path[:0] = [str(code / 'src'), str(code / 'tests'), str(site), str(shared)]
    root = code / 'outputs/robotwin-finalbank-offline-plan-20261031-r1'
    error = None
    try:
        if a.phase in ('tests', 'export', 'software-tests', 'source-admit',
                       'finalbank-train','finalbank-validation','finalbank-admission'):
            import torch
            if (Path(torch.__file__).resolve()!=Path(library_proof['torch_init']).resolve()
                    or sha(torch.__file__)!=library_proof['torch_init_sha256']):
                raise RuntimeError('Torch resolved outside the approved existing shared package')
            if torch.cuda.is_initialized():
                raise RuntimeError('CUDA was initialized in a CPU-only gate')
            def no_cuda(*args, **kwargs):
                raise RuntimeError('CUDA initialization forbidden in CPU gate')
            def no_step(*args, **kwargs):
                raise RuntimeError('Optimizer steps/fits forbidden in this compatibility gate')
            torch.cuda._lazy_init = no_cuda
            torch.optim.Adam.step = torch.optim.AdamW.step = no_step
            torch.set_num_threads(4)
        if a.phase.startswith('finalbank-'):
            finalbank_phase(a.phase.removeprefix('finalbank-'),ctx,code,out,allocation)
        elif a.phase in ('tests','software-tests'):
            os.environ['FB2_CPU_COMPAT_READER'] = ctx['chain_reader']
            os.environ['FB2_CPU_OWNED_HELPERS'] = ctx['owned_helpers']
            import pytest
            files=ctx.get('software_test_files') if a.phase=='software-tests' else [
                'tests/test_robotwin_history_corpus.py','tests/test_robotwin_offline_compat_gate.py',
                'tests/test_robotwin_cpu_residual_bootstrap.py','tests/test_robotwin_cpu_preflight_actual.py',
                'tests/test_robotwin_job_local_code.py']
            options=['--rootdir='+str(code),*['--deselect='+item for item in ctx.get('deselected_nodes',[])]] if a.phase=='software-tests' else []
            result = pytest.main(['-q','-p','no:cacheprovider',
                '--junitxml='+str(out/'fixtures.junit.xml'),*options,*[str(code/item) for item in files]])
            if result != 0:
                raise RuntimeError('Focused fixture suite failed: ' + str(result))
            suites = list(ET.parse(out / 'fixtures.junit.xml').getroot().iter('testsuite'))
            counts = {key:sum(int(s.attrib.get(key, '0')) for s in suites)
                      for key in ('tests','failures','errors','skipped')}
            if counts != dict(tests=ctx['expected_fixture_tests'], failures=0, errors=0, skipped=0):
                raise RuntimeError('Unexpected focused fixture coverage: ' + str(counts))
            (out / 'fixtures-summary.json').write_text(json.dumps(counts, indent=2) + '\n')
        elif a.phase=='source-admit':
            from fluffy_bassoon.meta_rl.robotwin_history_live import read_source_teacher_plan
            from fluffy_bassoon.meta_rl.robotwin_history_evaluate import admit_plan
            receipt=admit_plan(read_source_teacher_plan(code/ctx['teacher_plan'],ctx['teacher_plan_sha256']))
            if (receipt['source_iteration']!=100 or receipt['source_K4']!=600
                    or receipt['source_actions']!=1003790 or receipt['CUDA_generator_restored'] is not False
                    or receipt['additional_robot_actions']!=0):
                raise ValueError('Wrong source teacher CPU admission')
            (out/'source-admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
        elif a.phase == 'projection':
            project = load('prepared_byte_projection', root / 'project_corpus.py')
            sys.argv = [str(root / 'project_corpus.py'), '--purpose', 'reference',
                '--admission', ctx['reference_admission'], '--admission-sha256', ctx['reference_admission_sha256'],
                '--chain-reader', ctx['chain_reader'], '--output', str(out / 'private/projection'),
                '--expected-hostname', allocation['resource']['hostname']]
            project.main()
        elif a.phase == 'export':
            from fluffy_bassoon.meta_rl.robotwin_history_corpus import main as export_main
            snapshot = ctx['tip']['checkpoint']
            export_main(['--corpus-root', str(out / 'private/projection'),
                '--source-recovery', snapshot['path'], '--source-sha256', snapshot['sha256'],
                '--source-method', 'sac', '--teacher-recovery', snapshot['path'],
                '--teacher-sha256', snapshot['sha256'], '--teacher-method', 'sac',
                '--objectives', 'ad', 'dpt', '--partition', 'train',
                '--output', str(out / 'private/paired-export'), '--device', 'cpu', '--threads', '4',
                '--source-commit', ctx['code_commit'], '--expected-hostname', allocation['resource']['hostname']])
        else:
            import numpy as np
            index_path = out / 'private/paired-export/index.json'
            index = json.loads(index_path.read_text())
            verify_source65_index(index, ctx, out / 'private/projection/projection.json')
            for dataset in index['datasets'].values():
                if sha(dataset['path']) != dataset['sha256']:
                    raise ValueError('Export NPZ changed')
            with np.load(index['datasets']['ad']['path'], allow_pickle=False) as ad, \
                    np.load(index['datasets']['dpt']['path'], allow_pickle=False) as dpt:
                if set(ad.files) != set(dpt.files):
                    raise ValueError('Paired array schemas differ')
                for name in ad.files:
                    if name != 'target_noise' and not np.array_equal(ad[name], dpt[name]):
                        raise ValueError('Historical paired field changed: ' + name)
                if (not np.array_equal(ad['target_noise'], ad['executed_noise'])
                        or not np.isfinite(dpt['target_noise']).all()
                        or np.abs(dpt['target_noise']).max() > 1.500001
                        or int(ad['durations'].sum()) != 651638
                        or len(ad['durations']) != index['counts']['decisions']):
                    raise ValueError('Actual noise/label/mask-derived duration audit differs')
            receipt = dict(status='paired_CPU_compatibility_only', index_sha256=sha(index_path),
                source_rounds=65, counts=index['counts'], all_historical_paired_fields_equal=True,
                native_masks='Validated per decision by existing read_source; duration sum rechecked here',
                datasets=index['datasets'], selected_trial_prefix='included through390 globalK4 counter reconciliation',
                new_physical_actions=0, new_vla_queries=0, fits=0, SGD_updates=0,
                teacher_competence='not measured', source100='not admitted')
            (out / 'paired-audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        loaded = [name for name in sys.modules if forbidden(name)]
        torch = sys.modules.get('torch')
        cuda = torch is not None and torch.cuda.is_initialized()
        receipt = dict(phase=a.phase, error=error, seconds=time.monotonic() - started,
            denied_imports=guard.denied, forbidden_modules_loaded=loaded, cuda_initialized=cuda,
            shared_runtime=library_proof,torch_version=getattr(torch,'__version__',None),
            torch_origin=getattr(torch,'__file__',None),
            allowed_head_scope='Torch feature-steering heads only; no frozen VLA/native runtime',
            optimizer_steps_permitted=False)
        (out / (a.phase + '-imports.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        if loaded or cuda:
            raise RuntimeError('Forbidden module/CUDA state after phase')


if __name__ == '__main__':
    main()
