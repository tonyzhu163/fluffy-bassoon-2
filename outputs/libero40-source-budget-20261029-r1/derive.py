#!/usr/bin/env python3
"""Read-only source40 data/stat audit; no model, simulator, tests or pickle load.

--refresh reads completed JSON receipts and filesystem statistics over SSH.
Otherwise recompute projections from the compact input_snapshot in measurements.json.
The fixed-rate model is calibrated on one easy source task, not benchmark timing.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
RUNS = ('libero40-pi05-source-smoke-20261029-s00-train-r2',
        'libero40-pi05-source-smoke-20261029-s00-validation-r1')
TIB = 2**40
REMOTE = r'''
import hashlib,json,os,pathlib,time
root=pathlib.Path('/mowenjie/fluffy-bassoon-2/outputs')
result=[]
for name in ['libero40-pi05-source-smoke-20261029-s00-train-r2','libero40-pi05-source-smoke-20261029-s00-validation-r1']:
 p=root/name;files=[];stack=[p]
 while stack:
  d=stack.pop()
  for e in os.scandir(d):
   t=e.stat(follow_symlinks=False)
   if e.is_dir(follow_symlinks=False):stack.append(pathlib.Path(e.path))
   elif e.is_file(follow_symlinks=False):files.append(dict(path=str(pathlib.Path(e.path).relative_to(p)),bytes=t.st_size,allocated_bytes=t.st_blocks*512,mtime_ns=t.st_mtime_ns))
 inputs={}
 for filename in ['experiment/index.json','experiment/contract.json','schedule.json','gpu-probe.json']:
  b=(p/filename).read_bytes();inputs[filename]=dict(sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
 journals={}
 for e in files:
  filename=e['path']
  if filename.startswith('experiment/raw/decision-journal/') and filename.endswith('.json'):
   journals[filename]=json.loads((p/filename).read_text())
 v=os.statvfs(p)
 result.append(dict(run=name,root=str(p),captured_unix_ns=time.time_ns(),exit_code=int((p/'exit-code.txt').read_text()),files=files,inputs=inputs,journals=journals,free_bytes=v.f_bavail*v.f_frsize))
print(json.dumps(result))
'''


def grouped_storage(files):
    groups = collections.defaultdict(lambda: dict(files=0, bytes=0, allocated_bytes=0))
    for entry in files:
        parts = Path(entry['path']).parts
        if 'decision-journal' in parts:
            group = 'journal'
        elif len(parts) > 2 and parts[1] == 'raw':
            group = 'raw_episodes'
        elif len(parts) > 2 and parts[1].startswith('prefix-'):
            group = 'paired_prefixes'
        elif entry['path'].endswith('.pkl'):
            group = 'checkpoints'
        else:
            group = 'other'
        for key in ('bytes', 'allocated_bytes'):
            groups[group][key] += entry[key]
        groups[group]['files'] += 1
    return dict(groups)


def timing(snapshot):
    index = snapshot['inputs']['experiment/index.json']['value']
    files = {e['path']: e for e in snapshot['files']}
    def stamp(filename):
        return int(Path(filename).stem.split('-')[-1])
    decisions = sorted((stamp(n), n, v) for n, v in snapshot['journals'].items()
                       if Path(n).name.startswith('decision-'))
    resets = sorted((stamp(n), n, v) for n, v in snapshot['journals'].items()
                    if Path(n).name.startswith('reset-episode-'))
    episodes, cursor, previous_updates = [], 0, 0
    for ordinal, ((reset_ns, _, _), receipt) in enumerate(zip(resets, index['episode_receipts'])):
        selected = []
        while cursor < len(decisions):
            selected.append(decisions[cursor]); cursor += 1
            if selected[-1][2]['done']:
                break
        assert len(selected) == receipt['chunks']
        assert sum(d[2]['primitive_steps'] for d in selected) == receipt['primitive_steps']
        assert all(d[2]['case_id'] == receipt['case_id'] for d in selected)
        last_ns = files[selected[-1][1]]['mtime_ns']
        end_ns = resets[ordinal + 1][0] if ordinal + 1 < len(resets) else files['experiment/index.json']['mtime_ns']
        updates = receipt['source_updates'] - previous_updates
        previous_updates = receipt['source_updates']
        episodes.append(dict(episode=ordinal, chunks=receipt['chunks'], primitive_steps=receipt['primitive_steps'],
            updates=updates, collection_window_seconds=(last_ns-reset_ns)/1e9,
            post_episode_gap_seconds=(end_ns-last_ns)/1e9, case_boundary=(ordinal+1)%4 == 0))
    assert cursor == len(decisions) and len(episodes) == len(resets) == len(index['episode_receipts'])
    floor = statistics.median(e['post_episode_gap_seconds'] for e in episodes if not e['updates'])
    updated = [e for e in episodes if e['updates']]
    steady = updated[1:]
    rate = sum(max(0, e['post_episode_gap_seconds']-floor) for e in steady)/sum(e['updates'] for e in steady)
    result = dict(episodes=episodes, worker_seconds=index['elapsed_seconds'],
        collection_window_seconds=sum(e['collection_window_seconds'] for e in episodes),
        post_episode_gap_seconds=sum(e['post_episode_gap_seconds'] for e in episodes),
        no_update_post_gap_median_seconds=floor,
        approximate_steady_excess_seconds_per_update=rate, first_update_episode=updated[0])
    result['unattributed_startup_seconds'] = result['worker_seconds']-result['collection_window_seconds']-result['post_episode_gap_seconds']
    result['collection_seconds_per_chunk'] = result['collection_window_seconds']/index['chunks']
    # Probe-to-exit spans launcher preflight + worker; excludes scheduler startup/queue.
    probe = snapshot['inputs']['gpu-probe.json']['value']
    probe_epoch = datetime.datetime.fromisoformat(probe['created_at_utc']).timestamp()
    result['probe_to_exit_seconds'] = files['exit-code.txt']['mtime_ns']/1e9-probe_epoch
    result['probe_to_exit_minus_worker_seconds'] = result['probe_to_exit_seconds']-result['worker_seconds']
    return result


def compact(snapshot):
    i = snapshot['inputs']['experiment/index.json']['value']
    assert snapshot['exit_code'] == 0 and i['completed'] and i['smoke_qualified']
    files = {e['path']: e for e in snapshot['files']}
    receipts = i['episode_receipts']
    points, total = [], 0
    for ordinal in range(0, len(receipts), 4):
        total += sum(r['chunks'] for r in receipts[ordinal:ordinal+4])
        filename = f'experiment/case-{ordinal//4:05d}-source.pkl'
        if filename in files:
            points.append(dict(completed_cases=ordinal//4+1, transitions=total, bytes=files[filename]['bytes']))
    pub = i['publications'][-1]
    prefix_dir = Path('experiment')/Path(pub['datasets']['ad']['path']).parent
    prefix_bytes = sum(e['bytes'] for e in snapshot['files'] if Path(e['path']).parent == prefix_dir)
    probe = snapshot['inputs']['gpu-probe.json']['value']
    state_ids = sorted({r['initial_state'] for r in receipts})
    return dict(run=snapshot['run'], root=snapshot['root'], captured_unix_ns=snapshot['captured_unix_ns'],
        exit_code=snapshot['exit_code'], free_bytes=snapshot['free_bytes'],
        input_sha256={n: x['sha256'] for n, x in snapshot['inputs'].items()},
        source_contract={k: v for k, v in snapshot['inputs']['experiment/contract.json']['value'].items() if k not in ('cases', 'task')},
        source_task={k: v for k, v in snapshot['inputs']['experiment/contract.json']['value']['task'].items() if k != 'partitions'},
        scheduled_case_count=len(snapshot['inputs']['experiment/contract.json']['value']['cases']),
        gpu=dict(status=probe['status'], created_at_utc=probe['created_at_utc'],
            timing_comparison_eligible=probe['timing_comparison_eligible'],
            device_name=probe['devices'][0]['cuda_name'], device_total_bytes=probe['devices'][0]['cuda_total_bytes']),
        actual=dict(task_key=i['task_key'], partition=i['partition'], completed=i['completed'], episodes=len(receipts),
            successes=sum(r['success'] for r in receipts), seeded_cases=len({r['case_id'] for r in receipts}),
            distinct_initial_state_ids=state_ids, distinct_nominal_state_vector_hashes=len({r['initial_state_vector_sha256'] for r in receipts}),
            distinct_initial_observation_hashes=len({r['initial_observation_sha256'] for r in receipts}),
            chunks=i['chunks'], primitive_steps=i['actual_primitive_steps'], settling_steps=i['actual_settling_steps'],
            updates=i['source_updates'], completed_source_states_not_all_scheduled_cases=True),
        storage_groups=grouped_storage(snapshot['files']), checkpoint_samples=points,
        latest_recovery_bytes=files['experiment/source-recovery.pkl']['bytes'],
        latest_paired_prefix_bytes=prefix_bytes, latest_paired_prefix_transitions=pub['transitions'],
        timing=timing(snapshot))


def checkpoint_fit(points):
    xs=[p['transitions'] for p in points]; ys=[p['bytes'] for p in points]
    mx=statistics.mean(xs); my=statistics.mean(ys)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    return dict(fixed_bytes=my-slope*mx, bytes_per_transition=slope)


def cases_to_publish(cases, cadence):
    result=list(range(cadence, cases+1, cadence))
    if not result or result[-1] != cases:
        result.append(cases)
    return result


def conservative_schedule_bytes(cases, cadence):
    """Same static arithmetic as1650ae2 vla_source40_schedule.partition_budget.

    Complete_cases,400 primitive horizon,10/chunk,4episodes/case.
    Includes two recovery copies for atomic replacement and10%reserve.
    """
    chunks=cases*4*40
    pub=cases_to_publish(cases,cadence)
    rows=sum(n*4*40 for n in pub)
    raw=chunks*2*1024**2
    paired=rows*40*1024
    ckpt=(len(pub)+2)*256*1024**2+(rows+2*chunks)*32*1024
    return dict(raw_trace_and_journal_bytes=raw, paired_prefix_dataset_bytes=paired,
                source_checkpoint_and_replay_bytes=ckpt,
                total_bytes_with_ten_percent_reserve=math.ceil((raw+paired+ckpt)*1.10))


def project(snapshot, train=40, validation=20, cadence=16):
    source=snapshot[0]; a=source['actual']; groups=source['storage_groups']; t=source['timing']
    fitted=checkpoint_fit(source['checkpoint_samples'])
    raw_rate=(groups['raw_episodes']['bytes']+groups['journal']['bytes'])/a['chunks']
    prefix_rate=source['latest_paired_prefix_bytes']/source['latest_paired_prefix_transitions']
    observed_chunks=a['chunks']/a['episodes']; observed_steps=a['primitive_steps']/a['episodes']
    steady=t['approximate_steady_excess_seconds_per_update']; floor=t['no_update_post_gap_median_seconds']
    first=t['first_update_episode']
    cold=max(0, first['post_episode_gap_seconds']-floor-first['updates']*steady)
    overhead=max(s['timing']['probe_to_exit_minus_worker_seconds'] for s in snapshot)
    base=dict(tasks=40, task_partitions=80, episodes_per_case=4, publish_every_cases=cadence,
        train_seeded_cases_per_task=train, validation_seeded_cases_per_task=validation,
        train_distinct_state_ids_per_task=min(train,40), validation_distinct_state_ids_per_task=min(validation,10),
        train_additional_seeded_cases_on_repeated_state_ids=max(0,train-40),
        validation_additional_seeded_cases_on_repeated_state_ids=max(0,validation-10),
        immutable_train_publication_case_numbers=cases_to_publish(train,cadence),
        immutable_validation_publication_case_numbers=cases_to_publish(validation,cadence),
        requested_cases=40*(train+validation), episodes=40*(train+validation)*4)
    base['maximum_chunks']=base['episodes']*40
    base['maximum_primitive_steps']=base['episodes']*400
    base['maximum_settling_steps_with_one_qualification_reset_per_partition']=10*(base['episodes']+80)
    allowance={k:40*sum(conservative_schedule_bytes(c,cadence)[k] for c in (train,validation))
               for k in conservative_schedule_bytes(train,cadence)}
    base['conservative_schedule_storage_allowance']=allowance
    profiles={}
    for name,ce,pe in [('easy_smoke_rate',observed_chunks,observed_steps),('all_400_step_timeouts',40.,400.)]:
        chunks=base['episodes']*ce; ckpt=prefix=updates=0.
        for count in (train,validation):
            n=count*4*ce
            pubs=cases_to_publish(count,cadence)
            # Full latest recovery is retained; temporary atomic copy separately.
            ckpt+=sum(fitted['fixed_bytes']+fitted['bytes_per_transition']*p*4*ce for p in pubs)
            ckpt+=fitted['fixed_bytes']+fitted['bytes_per_transition']*n
            prefix+=sum(prefix_rate*p*4*ce for p in pubs if p*4*ce>500)
            updates+=20*max(0,n-500)
        ckpt*=40;prefix*=40;updates*=40
        raw=raw_rate*chunks
        seconds=(80*(t['unattributed_startup_seconds']+cold+overhead)
                 +chunks*t['collection_seconds_per_chunk']+base['episodes']*floor+updates*steady)
        # Strict update-count allowance is20times every queried chunk, including
        # warmup; conservative with respect to the actual >500 activation gate.
        update_ceiling=20*chunks
        count_conservative_seconds=seconds+(update_ceiling-updates)*steady
        temp=80*fitted['fixed_bytes']+40*fitted['bytes_per_transition']*(train+validation)*4*ce
        profiles[name]=dict(chunks=chunks, primitive_steps=base['episodes']*pe,
            approximate_source_optimizer_updates=updates, source_optimizer_update_allowance=update_ceiling,
            raw_and_journal_bytes=raw, immutable_plus_latest_checkpoint_bytes=ckpt,
            paired_cumulative_prefix_bytes=prefix, retained_bytes=raw+ckpt+prefix,
            temporary_atomic_recovery_allowance_bytes=temp,
            peak_bytes_including_all_atomic_recovery_copies=raw+ckpt+prefix+temp,
            modeled_gpu_hours=seconds/3600,
            conservative_update_count_gpu_hours=count_conservative_seconds/3600,
            planning_gpu_hours_with_1_5_uncertainty=count_conservative_seconds/3600*1.5)
    base['profiles']=profiles
    base['free_bytes_at_check']=min(s['free_bytes'] for s in snapshot)
    base['free_minus_conservative_schedule_allowance_bytes']=base['free_bytes_at_check']-allowance['total_bytes_with_ten_percent_reserve']
    base['model']=dict(checkpoint_linear_fit=fitted, raw_plus_journal_bytes_per_transition=raw_rate,
        paired_prefix_bytes_per_transition=prefix_rate, observed_chunks_per_episode=observed_chunks,
        observed_primitive_steps_per_episode=observed_steps, steady_mixed_excess_seconds_per_update=steady,
        first_update_cold_excess_seconds=cold, launcher_overhead_seconds_per_partition=overhead,
        retention='Immutable full checkpoints and cumulative AD/DPT prefixes every16cases plus final; latest full recovery everyepisode; raw and write-ahead journal retained.',
        limitations=['Single easy source task; task difficulty, image compression and reset cost differ.',
            'Mixed journal gaps include optimizer, synchronization, checkpoint and publication IO, not isolated GPU kernel timing.',
            'Use of older every-case IO in the cadence16 projection is conservative for collection IO.',
            'Warmup500 approximation omits boundary differences; separate20times-chunk update allowance covers them.',
            '40/20cases do not guarantee crossing500chunks on a task solved in very few decisions.',
            'No controller training, LIBERO90/Plus/Pro evaluation, new environment setup, scheduler queue or repeated qualification retries included.',
            '1.5time factor and static storage allowance are planning assumptions, not guarantees.'])
    return base


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    path=HERE/'measurements.json'
    if args.refresh:
        result=subprocess.run(['ssh','-p','4952','-o','BatchMode=yes','-o','ConnectTimeout=12',
            'root@10.200.0.11','python3 -'],input=REMOTE,text=True,check=True,capture_output=True)
        snapshots=[compact(s) for s in json.loads(result.stdout)]
    else:
        snapshots=json.loads(path.read_text())['input_snapshot']
    result=dict(format='libero40_source_runtime_storage_budget_v2',input_snapshot=snapshots,
        proposal=project(snapshots), alternatives=[project(snapshots,80,20),project(snapshots,32,20)])
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    p=result['proposal']
    print(json.dumps(dict(episodes=p['episodes'],primitive_ceiling=p['maximum_primitive_steps'],
        storage_allowance_TiB=p['conservative_schedule_storage_allowance']['total_bytes_with_ten_percent_reserve']/TIB,
        free_TiB=p['free_bytes_at_check']/TIB,
        profiles={k:dict(gpu_hours=v['modeled_gpu_hours'],planning_gpu_hours=v['planning_gpu_hours_with_1_5_uncertainty'],
            retained_TiB=v['retained_bytes']/TIB,peak_TiB=v['peak_bytes_including_all_atomic_recovery_copies']/TIB)
            for k,v in p['profiles'].items()}),indent=2))

if __name__=='__main__':
    main()
