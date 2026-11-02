"""Practice-only histories with isolated query branches; frozen original runtime."""
import argparse
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time

ROOT=Path('/mowenjie/fluffy-bassoon-2')
CORE=ROOT/'worktrees/opendrawer-ad-dpt-20261102-r1'
CORE_PIN='68f2c5aa33218709453a7821f6c6223ac3803035'
OLD=ROOT/'worktrees/opendrawer-paired-eval-20261102-r1'
CODE=Path(__file__).resolve().parents[2]
METHOD=os.environ['TASK_CURVE_METHOD']
assert METHOD in ('ad_full','dpt_full')
OUT=Path(os.environ['TASK_CURVE_OUTPUT'])
os.environ['TASK_READOUT_METHODS']=json.dumps([METHOD])
BASE=runpy.run_path(str(OLD/'outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py'))
PAIRING=BASE['PAIRING'];sha=BASE['sha'];ref=BASE['ref'];read_ref=BASE['read_ref'];write=BASE['write']
native_case=BASE['native_case'];motor_seeds=BASE['motor_seeds'];horizon=BASE['horizon']
SOURCE=ROOT/'outputs'/('opendrawer-ad-query-recovery-20261102-r1' if METHOD=='ad_full'
                       else 'opendrawer-dpt-full-query-recovery-20261102-r1')
TASKS=[(c,i) for c in ('OOD','ID') for i in range(50)]
WORLD=8


def fork_actor(actor):
    from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
    assert not actor.controller.pending and actor.controller.previous_done
    branch=HistoryResidualActor(actor.policy,mode='full',reset_each_episode=False)
    branch.controller=copy(actor.controller)
    branch.controller.state={k:v.detach().clone() for k,v in actor.controller.state.items()}
    branch.controller.previous_noise=actor.controller.previous_noise.copy()
    return branch


def memory_signature(actor):
    h=hashlib.sha256();c=actor.controller
    for k,v in sorted((c.state or {}).items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(c.previous_noise.tobytes())
    h.update(json.dumps([c.previous_reward,c.previous_done,c.episode_start,c.pending]).encode())
    return h.hexdigest()


def replay(actor,rows):
    for row in rows:
        actor.controller.act(row['features'],deterministic=True)
        actor.controller.observe(row['control'],row['reward'],row['done'])


def make_journal(rank):
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    # Use the exact frozen runtime API. Its legacy recording guard exceeds
    # this entire fixed cohort, including conservative constructor intents.
    journal=ActionJournal(OUT/'accounting',rank=rank,world=WORLD,attempt=os.environ['TASK_ATTEMPT'])
    maximum=sum(8*(horizon(c)+128) for c,i in TASKS[rank::WORLD])
    assert maximum<journal.limit
    return journal


def bind():
    import torch
    from fluffy_bassoon.meta_rl.vla_history import load_policy
    assert sys.platform=='linux' and not torch.cuda.is_available()
    assert BASE['head'](CORE)==CORE_PIN and BASE['clean'](CORE) and BASE['clean'](CODE)
    b=json.loads((SOURCE/'binding.json').read_text());assembly=read_ref(b['assembly'])
    selected=b['selected'][METHOD];assert read_ref(selected['status'])['converged']
    policy,manifest=load_policy(selected['candidate']['path'],expected_sha256=selected['candidate']['sha256'],device='cpu')
    contract=manifest['contract'];family=METHOD.split('_')[0]
    assert contract['source_commit']==CORE_PIN and contract['mode']=='full'
    assert contract['train_sha256']==assembly['datasets'][family+'_train']['sha256']
    assert contract['validation_sha256']==assembly['datasets'][family+'_validation']['sha256']
    assert policy.config.context_length==128
    reuse={}
    for c,i in TASKS:
        case={}
        for key,shot,kind in [('K0','K0','query'),('support1','K1','support'),('query1','K1','query')]:
            p=SOURCE/'episodes'/c/f'case-{i:03d}'/METHOD/shot/kind/'result.json'
            r=json.loads(p.read_text());assert r['complete'] and r['candidate']==selected['candidate']
            assert r['native_case']==native_case(c,i) and r['motor_seeds']==motor_seeds(c,i,kind=='support')
            assert r['initial_pairing']['report']['matched']
            read_ref(r['truth']);assert sha(r['records']['path'])==r['records']['sha256']
            case[key]=ref(p)
        assert len({read_ref(v)['initial_pairing']['canonical']['sha256'] for v in case.values()})==1
        reuse[f'{c}-{i:03d}']=case
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'binding.json').exists()
    write(OUT/'binding.json',dict(method=METHOD,cases_per_cohort=50,world=WORLD,
        driver_commit=BASE['head'](CODE),runtime_core=CORE_PIN,source_binding=ref(SOURCE/'binding.json'),
        original_driver=ref(OLD/'outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py'),
        assembly=b['assembly'],selected=selected,pairing=b['pairing'],reuse=reuse,
        query_K=[0,1,2,3,4,5],practice_K=[1,2,3,4,5],new_physical_episodes=800,
        new_control_upper_bound=sum(8*(horizon(c)+128) for c,i in TASKS),
        query_feedback_enters_practice=False,weights_frozen=True,training_window=64,context_capacity=128))
    print(json.dumps(ref(OUT/'binding.json')),flush=True)


def worker(rank,b,deadline):
    import imageio.v2 as imageio
    import torch
    from hydra import compose,initialize_config_dir
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
    from fluffy_bassoon.meta_rl.robocasa_residual import FrozenRoboCasaMotor,RoboCasaResidualEnvironment,collect_pending
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash
    from fluffy_bassoon.meta_rl.vla_history import load_policy
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from fluffy_bassoon.rl.action_residual import ResidualBound
    import multiprocessing as mp
    assert torch.cuda.device_count()==1;verify_selected_cuda(torch);torch.set_num_threads(4)
    mp.set_start_method('spawn',force=True)
    journal=make_journal(rank)
    assembly=read_ref(b['assembly']);init=assembly['contract']['initializer']
    profile=json.loads((ROOT/'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2/profile.json').read_text())
    assert BASE['head'](os.environ['RLINF_PATH'])==profile['rlinf_execution_commit']
    with initialize_config_dir(config_dir=str(CORE/'configs/native_ppo'),version_base=None):
        cfg=compose(config_name='robocasa365_pi05_eval')
    cfg.rollout.model.model_path=init['model_directory'];cfg.rollout.model.openpi_data.norm_stats_path=init['norm_path']
    assert sha(init['norm_path'])==init['normalization_sha256']
    from rlinf.models.embodiment.openpi import get_model
    model=get_model(cfg.rollout.model,torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats=json.loads(Path(init['norm_path']).read_text());motor=FrozenRoboCasaMotor(model,stats.get('norm_stats',stats),seed=991)
    bound=ResidualBound(**assembly['contract']['bound'])
    policy,_=load_policy(b['selected']['candidate']['path'],expected_sha256=b['selected']['candidate']['sha256'],device='cuda')
    assert policy.config.context_length==128
    weights=tensor_hash(policy.state_dict())

    def episode(c,i,k,kind,actor):
        directory=OUT/'episodes'/c/f'case-{i:03d}'/f'K{k}'/kind;p=directory/'result.json'
        if p.exists():
            r=json.loads(p.read_text());assert r['candidate']==b['selected']['candidate']
            rows=torch.load(r['records']['path'],map_location='cpu',weights_only=False)
            if kind=='support':replay(actor,rows)
            return r
        assert time.time()+300<deadline,'Operational checkpoint; preserve complete and partial episodes'
        directory.mkdir(parents=True,exist_ok=False)
        entry=dict(signature=memory_signature(actor),cache_length=int(actor.controller.state['length'][0]),
                   previous_reward=actor.controller.previous_reward,previous_done=actor.controller.previous_done)
        assert entry['cache_length']==(k-1 if kind=='support' else k)*(horizon(c)//50)
        class Paired(RoboCasaResidualEnvironment):
            def reset(self):
                obs=super().reset();physical=self.native.env.get_env_attr('_native_pairing')[0]
                snapshot=PAIRING['public_snapshot'](obs,native_case(c,i),physical)
                atomic_save(directory/'initial.pt',snapshot)
                marker,report=PAIRING['canonical_case'](Path(b['pairing']['directory'])/c/f'case-{i:03d}',snapshot,
                    save=atomic_save,load=lambda p:torch.load(p,map_location='cpu',weights_only=False),
                    source=dict(driver=b['driver_commit'],attempt=os.environ['TASK_ATTEMPT'],method=METHOD,shot=k,kind=kind))
                self.initial_pairing=dict(canonical=ref(marker),report=report,snapshot=ref(directory/'initial.pt'))
                return obs
        env=Paired(PAIRING['pairing_native_factory'](cfg.env.eval),[native_case(c,i)],horizon=horizon(c),
            output=directory/'videos',monitoring_root=directory/'monitor',journal=journal,writer_factory=imageio.get_writer,
            episode_id_prefix=f'{METHOD}-{c}-{i}-K{k}-{kind}-',rank=rank)
        seeds=motor_seeds(c,i,k if kind=='support' else 0)
        try:rows,episodes=collect_pending(actor,bound,motor,env,episodes=1,generator=None,deterministic=True,planned_motor_seeds=seeds)
        finally:env.close()
        assert len(rows)==horizon(c)//50 and len(episodes)==1
        truth_path=directory/'monitor'/(episodes[0]['id']+'.json');truth=json.loads(truth_path.read_text())
        success=int(truth['simulator_final_success']);rows[-1]['reward']=float(success)
        actor.install_terminal_reward(success);atomic_save(directory/'records.pt',rows)
        r=dict(complete=True,cohort=c,index=i,method=METHOD,K=k,kind=kind,success=success,candidate=b['selected']['candidate'],
            native_case=native_case(c,i),motor_seeds=seeds,horizon=horizon(c),records=ref(directory/'records.pt'),
            truth=ref(truth_path),initial_pairing=env.initial_pairing,training_source=CORE_PIN,
            evaluation_source=b['driver_commit'],memory_entry=entry)
        if kind=='support':write(p,r)
        print(json.dumps(dict(cohort=c,index=i,K=k,kind=kind,success=success,charged_actions=journal.charged())),flush=True)
        return r

    for c,i in TASKS[rank::WORLD]:
        reuse=b['reuse'][f'{c}-{i:03d}'];actor=HistoryResidualActor(policy,mode='full',reset_each_episode=False)
        first=read_ref(reuse['support1']);rows=torch.load(first['records']['path'],map_location='cpu',weights_only=False)
        replay(actor,rows);assert actor.controller.previous_reward==first['success']
        for k in range(2,6):
            episode(c,i,k,'support',actor)
            before=memory_signature(actor);branch=fork_actor(actor)
            assert memory_signature(branch)==before
            query_path=OUT/'episodes'/c/f'case-{i:03d}'/f'K{k}'/'query/result.json'
            reused_query=query_path.exists()
            query=episode(c,i,k,'query',branch)
            after=memory_signature(actor);assert after==before,'Scored query changed practice history'
            if reused_query:
                assert query['practice_memory_guard']['before']==before
            else:
                query['practice_memory_guard']=dict(before=before,after=after,unchanged=True,query_isolated=True)
                write(query_path,query)
        write(OUT/'trials'/c/f'case-{i:03d}.json',dict(complete=True,method=METHOD,reuse=reuse,source=b['driver_commit']))
    assert tensor_hash(policy.state_dict())==weights,'Frozen weights or normalization buffers changed'
    write(OUT/'workers'/f'rank-{rank:02d}.json',dict(completed=True,rank=rank,binding_sha256=sha(OUT/'binding.json'),
        source=b['driver_commit'],weights_unchanged=True,charged_actions=journal.charged()))


def summarize(b):
    import torch
    for rank in range(WORLD):
        w=json.loads((OUT/'workers'/f'rank-{rank:02d}.json').read_text())
        assert w['completed'] and w['weights_unchanged'] and w['binding_sha256']==sha(OUT/'binding.json')
    canonical_cache={}
    def verify(r,c,i,k,kind):
        assert r['complete'] and r['candidate']==b['selected']['candidate'] and r['method']==METHOD
        assert r['native_case']==native_case(c,i) and r['motor_seeds']==motor_seeds(c,i,k if kind=='support' else 0)
        assert r['training_source']==CORE_PIN and r['horizon']==horizon(c)
        assert r['kind']==kind
        assert sha(r['records']['path'])==r['records']['sha256']
        truth=read_ref(r['truth']);assert truth['case']==native_case(c,i) and int(truth['simulator_final_success'])==r['success']
        a=r['initial_pairing'];marker=read_ref(a['canonical']);key=a['canonical']['sha256']
        if key not in canonical_cache:
            p=Path(a['canonical']['path']).parent/'initial.pt';assert sha(p)==marker['snapshot_sha256']
            canonical_cache[key]=torch.load(p,map_location='cpu',weights_only=False)
        assert sha(a['snapshot']['path'])==a['snapshot']['sha256']
        report=PAIRING['compare_native_pair'](canonical_cache[key],torch.load(a['snapshot']['path'],map_location='cpu',weights_only=False))
        assert report==a['report'] and report['matched']
        if k>=2:
            assert r['evaluation_source']==b['driver_commit'] and r['K']==k
            assert r['memory_entry']['cache_length']==(k-1 if kind=='support' else k)*(horizon(c)//50)
            if kind=='query':
                g=r['practice_memory_guard'];assert g['unchanged'] and g['query_isolated'] and g['before']==g['after']==r['memory_entry']['signature']
        else:
            assert r['shot']==f'K{k}' and r['evaluation_source']==read_ref(b['source_binding'])['evaluation_driver_commit']
        return int(r['success'])
    cohorts={}
    for c in ('ID','OOD'):
        query={str(k):[] for k in range(6)};support={str(k):[] for k in range(1,6)}
        for i in range(50):
            reuse=b['reuse'][f'{c}-{i:03d}']
            query['0'].append(verify(read_ref(reuse['K0']),c,i,0,'query'))
            support['1'].append(verify(read_ref(reuse['support1']),c,i,1,'support'))
            query['1'].append(verify(read_ref(reuse['query1']),c,i,1,'query'))
            for k in range(2,6):
                for kind,dest in [('support',support),('query',query)]:
                    r=json.loads((OUT/'episodes'/c/f'case-{i:03d}'/f'K{k}'/kind/'result.json').read_text())
                    dest[str(k)].append(verify(r,c,i,k,kind))
        cohorts[c]=dict(cases=50,query_successes={k:sum(v) for k,v in query.items()},
            support_successes={k:sum(v) for k,v in support.items()},query_cases=query,support_cases=support,
            paired_vs_K0={k:BASE['paired'](v,query['0']) for k,v in query.items() if k!='0'},
            paired_K5_vs_K1=BASE['paired'](query['5'],query['1']))
    result=dict(completed=True,independent_model_complete=True,method=METHOD,binding=ref(OUT/'binding.json'),
        cohorts=cohorts,practice_only_memory=True,query_feedback_in_practice=False,new_episodes=800,
        single_source_task=True,single_seed=True,context_extrapolation_beyond_training_window=True,
        render_roundoff='Exact native physical/camera/public state; <=1uint8 image roundoff, original pixels',
        p_values='unadjusted exploratory exact paired tests; supports charged separately')
    write(OUT/'RESULT.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--summarize',action='store_true')
    p.add_argument('--worker',type=int);p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.bind:bind();return
    b=json.loads((OUT/'binding.json').read_text())
    assert b['method']==METHOD and b['runtime_core']==BASE['head'](CORE)==CORE_PIN
    assert b['driver_commit']==BASE['head'](CODE) and BASE['clean'](CORE) and BASE['clean'](CODE)
    assert b['original_driver']==ref(OLD/'outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py')
    if a.summarize:summarize(b);return
    if a.worker is not None:worker(a.worker,b,a.deadline);return
    import torch
    node=int(os.environ['RANK']);assert os.environ['WORLD_SIZE']=='4' and torch.cuda.device_count()==2
    attempt=Path(os.environ['TASK_ATTEMPT_OUT']);attempt.mkdir(parents=True,exist_ok=False)
    assert sha(OUT/'binding.json')==os.environ['TASK_BINDING_SHA']
    assert os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize>=64<<30
    probe=attempt/'gpu-probe.json'
    subprocess.run([sys.executable,'-B',str(CORE/'scripts/[train]gpu_probe.py'),'--expected-count','2',
        '--purpose','evaluation','--baige-container','--output',str(probe)],check=True)
    devices=json.loads(probe.read_text())['devices'];children=[];streams=[]
    write(attempt/'manifest.json',dict(node=node,hostname=socket.gethostname(),source=b['driver_commit'],
        runtime_core=CORE_PIN,binding=ref(OUT/'binding.json'),devices=devices,deadline=a.deadline))
    for local,d in enumerate(devices):
        rank=node*2+local;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d['visibility_token']),
            LYCO_BAIGE_GPU_MAP=str(probe),LYCO_BAIGE_ISOLATED_UUID=d['uuid'],MUJOCO_EGL_DEVICE_ID=str(d['allocation_ordinal']))
        stream=(attempt/f'worker-{rank:02d}.log').open('x');streams.append(stream)
        children.append(subprocess.Popen([sys.executable,'-B',__file__,'--worker',str(rank),'--deadline',str(a.deadline)],
            env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    while not all(c.poll() is not None for c in children):
        if any(c.poll() not in (None,0) for c in children):
            for c in children:
                if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
            break
        time.sleep(5)
    codes=[c.wait() for c in children]
    for stream in streams:stream.close()
    write(attempt/'workers-terminal.json',dict(exit_codes=codes,finished_unix=time.time()))
    if any(codes):raise RuntimeError('Practice curve failed; preserve all records before recovery')
    if node==0:
        future=runpy.run_path(str(CODE/'outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py'))
        future['wait_for_nodes'](attempt.parent,a.deadline)


if __name__=='__main__':main()
