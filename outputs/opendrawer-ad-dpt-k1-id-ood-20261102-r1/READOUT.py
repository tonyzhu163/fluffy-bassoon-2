"""Additional one-shot ID/OOD episodes; immutable original training runtime."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
TRAIN = ROOT/'outputs/opendrawer-ppo-ad-dpt-20261102-r1'
OUT = Path(os.environ.get('TASK_READOUT_OUTPUT',str(ROOT/'outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1')))
CORE = ROOT/'worktrees/opendrawer-ad-dpt-20261102-r1'
CORE_PIN = '68f2c5aa33218709453a7821f6c6223ac3803035'
CODE = Path(__file__).resolve().parents[2]
PAIRING_PATH = CODE/'src/fluffy_bassoon/meta_rl/robocasa_pairing.py'
PAIRING = runpy.run_path(str(PAIRING_PATH))
CASE_BANK = Path(os.environ.get('TASK_READOUT_CASE_BANK',str(ROOT/'outputs/opendrawer-ad-dpt-paired-cases-20261102-r1')))
ALL_METHODS = ('ppo_initial','ppo_final','ad_full','ad_static','dpt_full','dpt_static')
if 'TASK_READOUT_METHODS' not in os.environ:
    raise ValueError('Select explicit ready methods; evaluation has no all-model default barrier')
METHODS = tuple(json.loads(os.environ['TASK_READOUT_METHODS']))
assert METHODS and len(set(METHODS))==len(METHODS) and set(METHODS)<=set(ALL_METHODS)
STUDENTS = tuple(name for name in METHODS if not name.startswith('ppo_'))
FULL = tuple(name for name in METHODS if name.endswith('_full'))
FRESH_ID = os.environ.get('TASK_READOUT_FRESH_ID')=='1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def ref(path):
    return dict(path=str(path),sha256=sha(path))


def read_ref(value):
    assert sha(value['path'])==value['sha256'],value['path']
    return json.loads(Path(value['path']).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def head(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def clean(path):
    return not subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()


def tasks():
    return [('OOD',i) for i in range(50)]+[('ID',i) for i in range(0 if FRESH_ID else 8,50)]


def episode_count(cohort):
    return len(METHODS)+2*len(FULL) if cohort=='OOD' or FRESH_ID else 2*len(FULL)


def horizon(cohort):
    return 750 if cohort=='ID' else 900


def native_case(cohort,index):
    return dict(bank='heldout_ad_dpt' if cohort=='ID' else 'heldout_ad_dpt_ood',
        task='OpenDrawer' if cohort=='ID' else 'CloseFridge',split='target',
        native_seed=(880110000 if cohort=='ID' else 880210000)+index,
        reset_ordinal=0,slot=index,counted=True,
        layout_source='native_target_constructor_seed_first_reset')


def motor_seeds(cohort,index,support):
    import numpy as np
    start=881210000 if cohort=='ID' else 881310000
    return np.random.default_rng(start+index+10000*support).integers(0,2**63-1,horizon(cohort)//50).tolist()


def bind():
    import torch
    assert sys.platform=='linux' and not torch.cuda.is_available()
    assert head(CORE)==CORE_PIN and clean(CORE) and clean(CODE)
    assembly=json.loads((TRAIN/'data/assembly.json').read_text())
    assert FRESH_ID,'New pairing snapshots require fresh ID episodes; retain legacy records separately'
    selected={}
    # Only the requested model's own readiness can gate its evaluation.
    for name in STUDENTS:
        folder=TRAIN/'fits'/name;status=json.loads((folder/'training-status.json').read_text())
        assert status['converged'],name
        payload=torch.load(folder/'candidate.pt',map_location='cpu',weights_only=False)
        contract=payload['manifest']['contract'];objective,mode=name.split('_')
        assert contract['source_commit']==CORE_PIN and contract['objective']==objective and contract['mode']==mode
        assert contract['train_sha256']==assembly['datasets'][objective+'_train']['sha256']
        assert contract['validation_sha256']==assembly['datasets'][objective+'_validation']['sha256']
        assert contract['schedule']['until_converged'] and status['stop_reason']=='validation_plateau'
        assert payload['manifest']['validation_nll']==status['best_validation_nll']
        selected[name]=dict(candidate=ref(folder/'candidate.pt'),status=ref(folder/'training-status.json'),
            selected_update=payload['manifest']['update'],validation_nll=payload['manifest']['validation_nll'])
    bank_contract=dict(format=PAIRING['FORMAT'],max_RGB_delta=PAIRING['MAX_RGB_DELTA'],
        runtime_core_commit=CORE_PIN,pairing_code_sha256=sha(PAIRING_PATH),
        assembly=ref(TRAIN/'data/assembly.json'),ID=dict(seed_first=880110000,cases=50,horizon=750),
        OOD=dict(seed_first=880210000,cases=50,horizon=900),
        native_process_start_method='spawn',child_global_RNG_seed='native_seed',observations_modified=False)
    CASE_BANK.mkdir(parents=True,exist_ok=True)
    bank_path=CASE_BANK/'contract.json'
    if bank_path.exists():assert json.loads(bank_path.read_text())==bank_contract
    else:
        with bank_path.open('x') as stream:stream.write(json.dumps(bank_contract,indent=2)+'\n')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'binding.json';assert not path.exists()
    write(path,dict(completed=True,evaluation_driver_commit=head(CODE),runtime_core_commit=CORE_PIN,
        assembly=ref(TRAIN/'data/assembly.json'),selected=selected,initial=assembly['initial'],teacher=assembly['teacher'],
        primary_endpoint='one complete support episode -> separate query, frozen weights',
        methods=list(METHODS),fresh_ID=FRESH_ID,independent_model_admission=True,
        pairing=dict(contract=ref(bank_path),directory=str(CASE_BANK),code=ref(PAIRING_PATH)),
        ID=dict(task='OpenDrawer',cases=50,seed_first=880110000,horizon=750),
        OOD=dict(task='CloseFridge',cases=50,seed_first=880210000,horizon=900),
        source_training_unchanged=True,new_physical_episodes=sum(episode_count(c) for c,_ in tasks()),
        new_control_upper_bound=sum(episode_count(c)*(horizon(c)+128) for c,_ in tasks())))
    print(json.dumps(ref(path)),flush=True)


def check_plan():
    assignment=[tasks()[rank::8] for rank in range(8)]
    expected_tasks=100 if FRESH_ID else 92
    assert sum(map(len,assignment))==expected_tasks and len(set(tasks()))==expected_tasks
    controls=[sum(episode_count(cohort)*(horizon(cohort)+128) for cohort,_ in rows) for rows in assignment]
    assert max(controls)<125000
    assert all(native_case('ID',i)['task']=='OpenDrawer' and len(motor_seeds('ID',i,False))==15 for i in range(50))
    assert all(native_case('OOD',i)['task']=='CloseFridge' and len(motor_seeds('OOD',i,False))==18 for i in range(50))
    old=runpy.run_path(str(CORE/'outputs/opendrawer-ppo-ad-dpt-20261102-r1/RUN.py'))
    for i in range(50):
        assert native_case('ID',i)==old['case'](i)
        for support in (False,True):assert motor_seeds('ID',i,support)==old['noise_plan'](i,support)
    print(json.dumps(dict(completed=True,cohort='ID50/OOD50',methods=list(METHODS),fresh_ID=FRESH_ID,
        new_episodes=sum(episode_count(c) for c,_ in tasks()),controls_per_rank=controls,
        ID_cases_and_motor_draws_exactly_preserved=True)))


def worker(rank,binding,deadline):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from hydra import compose,initialize_config_dir
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
    from fluffy_bassoon.meta_rl.robocasa_residual import FrozenRoboCasaMotor,RoboCasaResidualEnvironment,collect_pending
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    from fluffy_bassoon.meta_rl.vla_history import load_policy
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from fluffy_bassoon.rl.action_residual import ResidualBound
    assert torch.cuda.device_count()==1
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    verify_selected_cuda(torch);torch.set_num_threads(4)
    assembly=read_ref(binding['assembly']);contract=assembly['contract'];initializer=contract['initializer']
    profile=json.loads((ROOT/'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2/profile.json').read_text())
    assert head(os.environ['RLINF_PATH'])==profile['rlinf_execution_commit']
    with initialize_config_dir(config_dir=str(CORE/'configs/native_ppo'),version_base=None):
        cfg=compose(config_name='robocasa365_pi05_eval')
    cfg.rollout.model.model_path=initializer['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path=initializer['norm_path']
    from rlinf.models.embodiment.openpi import get_model
    model=get_model(cfg.rollout.model,torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats=json.loads(Path(initializer['norm_path']).read_text())
    motor=FrozenRoboCasaMotor(model,stats.get('norm_stats',stats),seed=991)
    bound=ResidualBound(**contract['bound'])
    old=runpy.run_path(str(CORE/'outputs/opendrawer-ppo-ad-dpt-20261102-r1/RUN.py'))
    policies={name:old['load_ppo'](binding['initial'] if name=='ppo_initial' else binding['teacher'],'cuda')
        for name in METHODS if name.startswith('ppo_')}
    candidates=dict(ppo_initial=binding['initial'],ppo_final=binding['teacher'])
    for name in STUDENTS:
        selected=binding['selected'][name];assert read_ref(selected['status'])['converged']
        candidates[name]=selected['candidate']
        policy,manifest=load_policy(candidates[name]['path'],expected_sha256=candidates[name]['sha256'],device='cuda')
        assert manifest['contract']['source_commit']==CORE_PIN
        policies[name]=HistoryResidualActor(policy,mode=name.split('_')[1])
    journal=ActionJournal(OUT/'accounting',rank=rank,world=8,attempt=os.environ['TASK_ATTEMPT'])
    assigned=tasks()[rank::8]

    def episode(cohort,index,name,actor,shot,kind):
        directory=OUT/'episodes'/cohort/f'case-{index:03d}'/name/shot/kind
        result=directory/'result.json'
        if result.exists():
            value=json.loads(result.read_text());assert value['candidate']==candidates[name]
            assert sha(value['records']['path'])==value['records']['sha256']
            return value,torch.load(value['records']['path'],map_location='cpu',weights_only=False)
        assert time.time()+300<deadline,'Operational boundary; preserve completed cases'
        directory.mkdir(parents=True,exist_ok=False)
        class PairedEnvironment(RoboCasaResidualEnvironment):
            def reset(self):
                observation=super().reset()
                physical=self.native.env.get_env_attr('_native_pairing')[0]
                snapshot=PAIRING['public_snapshot'](observation,native_case(cohort,index),physical)
                atomic_save(directory/'initial.pt',snapshot)
                marker,report=PAIRING['canonical_case'](CASE_BANK/cohort/f'case-{index:03d}',snapshot,
                    save=atomic_save,load=lambda path:torch.load(path,map_location='cpu',weights_only=False),
                    source=dict(driver=binding['evaluation_driver_commit'],attempt=os.environ['TASK_ATTEMPT'],
                        method=name,shot=shot,kind=kind,contract=binding['pairing']['contract']))
                self.initial_pairing=dict(canonical=ref(marker),report=report,snapshot=ref(directory/'initial.pt'))
                return observation
        env=PairedEnvironment(PAIRING['pairing_native_factory'](cfg.env.eval),[native_case(cohort,index)],
            horizon=horizon(cohort),output=directory/'videos',monitoring_root=directory/'monitor',journal=journal,
            writer_factory=imageio.get_writer,episode_id_prefix=f'{cohort}-{index}-{name}-{shot}-{kind}-',rank=rank)
        support=kind=='support'
        try:
            rows,episodes=collect_pending(actor,bound,motor,env,episodes=1,generator=None,deterministic=True,
                verify_zero=cohort=='OOD' and index==assigned[0][1] and name=='ppo_initial',
                planned_motor_seeds=motor_seeds(cohort,index,support))
        finally:env.close()
        assert len(rows)==horizon(cohort)//50 and len(episodes)==1
        truth_path=directory/'monitor'/(episodes[0]['id']+'.json');truth=json.loads(truth_path.read_text())
        success=int(truth['simulator_final_success']);rows[-1]['reward']=float(success)
        atomic_save(directory/'records.pt',rows)
        value=dict(cohort=cohort,index=index,method=name,shot=shot,kind=kind,success=success,complete=True,
            candidate=candidates[name],native_case=native_case(cohort,index),motor_seeds=motor_seeds(cohort,index,support),
            initial_features_sha256=hashlib.sha256(np.asarray(rows[0]['features'],np.float32).tobytes()).hexdigest(),
            initial_base_sha256=hashlib.sha256(np.asarray(rows[0]['base_chunk'],np.float32).tobytes()).hexdigest(),
            records=ref(directory/'records.pt'),truth=ref(truth_path),horizon=horizon(cohort),
            initial_pairing=env.initial_pairing,
            training_source=CORE_PIN,evaluation_source=binding['evaluation_driver_commit'])
        write(result,value)
        print(json.dumps(dict(cohort=cohort,index=index,method=name,shot=shot,kind=kind,
            success=success,charged_actions=journal.charged())),flush=True)
        return value,rows

    for cohort,index in assigned:
        baseline={}
        if cohort=='OOD' or FRESH_ID:
            for name in METHODS:
                row,_=episode(cohort,index,name,policies[name],'K0','query');baseline[name]=row
            assert len({r['initial_pairing']['canonical']['sha256'] for r in baseline.values()})==1
        queries=[]
        for name in FULL:
            actor=HistoryResidualActor(policies[name].policy,mode='full',reset_each_episode=False)
            support,rows=episode(cohort,index,name,actor,'K1','support')
            actor.controller.reset_trial()
            for row in rows:
                actor.controller.act(row['features'],deterministic=True)
                actor.controller.observe(row['control'],row['reward'],row['done'])
            query,_=episode(cohort,index,name,actor,'K1','query');queries.append(query)
            if cohort=='OOD' or FRESH_ID:
                assert query['initial_pairing']['canonical']==baseline[name]['initial_pairing']['canonical']
        if queries:assert len({r['initial_pairing']['canonical']['sha256'] for r in queries})==1
    write(OUT/'workers'/f'rank-{rank:02d}.json',dict(completed=True,rank=rank,source=binding['evaluation_driver_commit'],
        binding_sha256=sha(OUT/'binding.json'),tasks=assigned,charged_actions=journal.charged()))


def paired(a,b):
    wins=sum(x>y for x,y in zip(a,b));losses=sum(x<y for x,y in zip(a,b));n=wins+losses
    p=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1.
    return dict(wins=wins,losses=losses,ties=len(a)-n,exact_p=p)


def summarize(binding):
    import torch
    for rank in range(8):assert json.loads((OUT/'workers'/f'rank-{rank:02d}.json').read_text())['completed']
    if not FRESH_ID:assert json.loads((TRAIN/'RESULT.json').read_text())['completed']
    candidates=dict(ppo_initial=binding['initial'],ppo_final=binding['teacher'])
    candidates.update({name:value['candidate'] for name,value in binding['selected'].items()})

    def verified(path,cohort,index,name,shot,kind):
        row=json.loads(path.read_text())
        assert row['complete'] and row['method']==name and row['candidate']==candidates[name]
        assert row.get('case_index',row.get('index'))==index
        assert row['native_case']==native_case(cohort,index)
        assert row['motor_seeds']==motor_seeds(cohort,index,kind=='support')
        record=row.get('records',dict(path=row.get('records_path'),sha256=row.get('records_sha256')))
        assert sha(record['path'])==record['sha256']
        truth=read_ref(row['truth'])
        assert truth['case']==native_case(cohort,index) and int(truth['simulator_final_success'])==row['success']
        assert row.get('horizon',row.get('primitive_policy_steps'))==horizon(cohort)
        initial=row['initial_pairing'];assert sha(initial['snapshot']['path'])==initial['snapshot']['sha256']
        canonical=read_ref(initial['canonical']);snapshot_path=Path(initial['canonical']['path']).parent/'initial.pt'
        assert sha(snapshot_path)==canonical['snapshot_sha256']
        report=PAIRING['compare_native_pair'](torch.load(snapshot_path,map_location='cpu',weights_only=False),
            torch.load(initial['snapshot']['path'],map_location='cpu',weights_only=False))
        assert report==initial['report'] and report['matched']
        return row

    tables={}
    for cohort in ('ID','OOD'):
        zero={name:[] for name in METHODS};one={name:[] for name in FULL};support={name:[] for name in FULL}
        for index in range(50):
            for name in METHODS:
                path=(TRAIN/'evaluation'/f'rank-{index%2:02d}'/'K0'/name/f'case-{index:03d}'/'query/result.json'
                    if cohort=='ID' and not FRESH_ID else OUT/'episodes'/cohort/f'case-{index:03d}'/name/'K0/query/result.json')
                row=verified(path,cohort,index,name,'K0','query');zero[name].append(row)
            assert len({zero[name][-1]['initial_pairing']['canonical']['sha256'] for name in METHODS})==1
            for name in FULL:
                directory=(TRAIN/'evaluation'/f'rank-{index%2:02d}'/'K1'/name/f'case-{index:03d}'
                    if cohort=='ID' and index<8 and not FRESH_ID else OUT/'episodes'/cohort/f'case-{index:03d}'/name/'K1')
                query=verified(directory/'query/result.json',cohort,index,name,'K1','query')
                previous=verified(directory/'support/result.json',cohort,index,name,'K1','support')
                assert query['candidate']==binding['selected'][name]['candidate']
                assert query['initial_pairing']['canonical']==zero[name][-1]['initial_pairing']['canonical']
                assert previous['initial_pairing']['canonical']==query['initial_pairing']['canonical']
                one[name].append(query);support[name].append(previous)
        controls={name:[row['success'] for row in rows] for name,rows in zero.items()}
        primary=dict(controls)
        for name,rows in one.items():primary[name]=[row['success'] for row in rows]
        tables[cohort]=dict(cases=50,primary_successes={name:sum(values) for name,values in primary.items()},
            zero_shot_successes={name:sum(values) for name,values in controls.items()},
            support_successes={name:sum(row['success'] for row in rows) for name,rows in support.items()},
            student_vs_PPO={f'{name}-vs-{baseline}':paired(primary[name],primary[baseline])
                for name in STUDENTS for baseline in ('ppo_initial','ppo_final') if baseline in primary},
            full_K1_vs_K0={name:paired(primary[name],controls[name]) for name in FULL},
            context_blind_controls='PPO/static K0 query results reused; support cannot affect their policy; no invented support episodes')
    result=dict(completed=True,primary_endpoint='one-shot full students vs frozen PPO/static query controls',
        binding=ref(OUT/'binding.json'),methods=list(METHODS),independent_phase_complete=True,
        ID=tables['ID'],OOD=tables['OOD'],
        OOD_meaning='CloseFridge held out from controller training, not necessarily foundation pretraining',
        single_source_task=True,single_seed=True,source_streams_span_different_scenes=True,
        pairing_contract=binding['pairing']['contract'],
        render_roundoff='Exact native scene/state/camera and public state; RGB differences at most1/255 per component; unmodified policy inputs; feature/base hashes need not be bitwise identical across resets',
        reward='simulator-reward development pilot, not Flora result')
    write(OUT/'RESULT.json',result)
    lines=['# One-shot ID/OOD comparison','','Full students use one complete support then a separate query. PPO/static are context-blind query controls.','',
        '| Policy | ID OpenDrawer /50 | OOD CloseFridge /50 |','| --- | --- | --- |']
    lines += [f'| {name} | {tables["ID"]["primary_successes"][name]} | {tables["OOD"]["primary_successes"][name]} |' for name in METHODS]
    lines += ['','Support success,zero-shot results and paired comparisons are in RESULT.json.','']
    (OUT/'RESULT.md').write_text('\n'.join(lines));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--check-plan',action='store_true')
    p.add_argument('--summarize',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--deadline',type=float)
    args=p.parse_args()
    if args.check_plan:check_plan();return
    if args.bind:bind();return
    binding=json.loads((OUT/'binding.json').read_text());assert head(CORE)==binding['runtime_core_commit']==CORE_PIN
    assert binding['pairing']['directory']==str(CASE_BANK)
    assert read_ref(binding['pairing']['contract'])['pairing_code_sha256']==sha(PAIRING_PATH)
    assert binding['pairing']['code']==ref(PAIRING_PATH)
    assert binding['methods']==list(METHODS) and binding['fresh_ID']==FRESH_ID
    assert head(CODE)==binding['evaluation_driver_commit'] and clean(CORE) and clean(CODE)
    if args.summarize:summarize(binding);return
    if args.worker is not None:worker(args.worker,binding,args.deadline);return
    import torch
    node=int(os.environ['RANK']);assert os.environ['WORLD_SIZE']=='4' and torch.cuda.device_count()==2
    attempt=Path(os.environ['TASK_ATTEMPT_OUT']);attempt.mkdir(parents=True,exist_ok=False)
    assert sha(OUT/'binding.json')==os.environ['TASK_BINDING_SHA']
    stat=os.statvfs(OUT);assert stat.f_bavail*stat.f_frsize>=64<<30
    write_probe=attempt/'write-probe';write_probe.write_text('write probe\n');write_probe.unlink()
    probe=attempt/'gpu-probe.json'
    subprocess.run([sys.executable,'-B',str(CORE/'scripts/[train]gpu_probe.py'),'--expected-count','2','--purpose','training',
        '--baige-container','--baige-exclusive-training','--output',str(probe)],check=True)
    devices=json.loads(probe.read_text())['devices']
    write(attempt/'manifest.json',dict(node=node,hostname=socket.gethostname(),source=head(CODE),runtime_core=CORE_PIN,
        binding=ref(OUT/'binding.json'),devices=devices,deadline=args.deadline,free_bytes=stat.f_bavail*stat.f_frsize))
    children=[];streams=[]
    for local,device in enumerate(devices):
        rank=node*2+local
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device['visibility_token']),LYCO_BAIGE_GPU_MAP=str(probe),
            LYCO_BAIGE_ISOLATED_UUID=device['uuid'],MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
        stream=(attempt/f'worker-{rank:02d}.log').open('x');streams.append(stream)
        children.append(subprocess.Popen([sys.executable,'-B',__file__,'--worker',str(rank),'--deadline',str(args.deadline)],
            env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    while not all(c.poll() is not None for c in children):
        if any(c.poll() not in (None,0) for c in children):
            for c in children:
                if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
            break
        time.sleep(5)
    codes=[c.wait() for c in children]
    for s in streams:s.close()
    write(attempt/'workers-terminal.json',dict(exit_codes=codes,finished_unix=time.time()))
    if any(codes):raise RuntimeError('Additional readout failed; preserve records and diagnose before retry')


if __name__=='__main__':main()
