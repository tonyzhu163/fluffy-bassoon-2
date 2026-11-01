#!/usr/bin/env python
"""Bounded native-only headroom selection using the existing RLinf evaluator.

One exclusive allocation, serial child waves, no optimizer and no scheduler API.
Raw episodes/journals/videos survive every stop or failure. Flora is not called.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import digest,sha,write_new
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog
from fluffy_bassoon.meta_rl.rlinf_libero_headroom import candidate_batches,headroom_schedule,evaluation_seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('profile','model','norm','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Native evaluation only inside the actual allocated Linux job')
    profile_path=Path(args.profile);profile=json.loads(profile_path.read_text());root=Path(args.output)
    source=Path(__file__).resolve().parents[2]
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if (head!=profile['source_commit'] or profile['allocated_gpu_count']!=4 or profile['world_size']!=4
            or not 0<profile['scheduler_wall_seconds']<=14340 or profile['closure_reserve_seconds']<300
            or profile['maximum_new_actions']!=375160 or profile['maximum_episodes']!=904
            or profile['maximum_output_bytes']!=32*(1<<30) or not 0<profile['maximum_child_seconds']<=900):
        raise ValueError('Concrete pinned four-GPU finite native search profile required')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'CHILDREN').exists() or (root/'search-summary.json').exists():
        raise ValueError('No blind resume/recollection or output reuse after a partial search')
    children=root/'CHILDREN';children.mkdir()
    records=[];charged=0;episodes=0;confirmations=0;reason='exhausted_candidates';selected=None
    log=ProgressLog(root/'progress','S',contract_sha256=sha(profile_path)).start()

    def bytes_used():
        # Only this fresh output identity, never shared corpora or model trees.
        return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())

    def emit(phase,**updates):
        view=json.loads(Path(profile['cost_status_path']).read_text())
        log.emit(phase=phase,physical_actions=charged,episodes_complete=episodes,
            aggregate_gpu_seconds=view['gpu_seconds_actual'],
            this_open_allocation_gpu_seconds=(time.time()-profile['allocated_start_unix'])*4,
            elapsed_seconds=time.time()-profile['allocated_start_unix'],
            remaining_wall_seconds=max(0,profile['deadline_unix']-time.time()),**updates)

    def run_wave(tasks,phase):
        nonlocal charged,episodes
        schedule=headroom_schedule(tasks,phase);maximum=len(schedule)*415
        if charged+maximum>375160 or episodes+len(schedule)>904:raise TimeoutError('Physical search ceiling before next wave')
        if time.time()+profile['maximum_child_seconds']+profile['closure_reserve_seconds']>=profile['deadline_unix']:
            raise TimeoutError('Finite allocation closure reserve before next wave')
        if bytes_used()+len(schedule)*(32<<20)+(64<<20)>profile['maximum_output_bytes']:
            raise TimeoutError('Bounded output reserve before next wave')
        ordinal=len(records);wave=children/f'{ordinal:03d}-{phase}-'+Path('unused') if False else children/f'{ordinal:03d}-{phase}'
        wave.mkdir()
        seed=evaluation_seed(tasks,phase)
        child=dict(profile,phase=phase,arm='S',headroom_tasks=list(tasks),eval_seed=seed,
            case_schedule_sha256=digest(schedule),maximum_new_actions=maximum,
            parent_profile=dict(path=str(profile_path),sha256=sha(profile_path)),
            child_ordinal=ordinal,additional_smoke_episodes=0,optimizer_updates=0)
        write_new(wave/'profile.json',child);write_new(wave/'cases.json',schedule)
        emit(phase,iteration=ordinal,headroom_tasks=list(tasks),labels_complete=0)
        command=[sys.executable,'-B',str(source/'scripts/diagnostics/rlinf_flora_ppo_entry.py'),
            '--phase',phase,'--arm','S','--world','4','--headroom-tasks',','.join(map(str,tasks)),
            '--eval-seed',str(seed),'--profile',str(wave/'profile.json'),'--model',args.model,'--norm',args.norm,
            '--output',str(wave/'native-output'),'--exchange',str(wave/'exchange'),
            '--monitoring',str(wave/'monitoring'),'--accounting',str(wave/'actions'),
            '--attempt',f'wave-{ordinal:03d}']
        write_new(wave/'command.json',command)
        try:
            with (wave/'console.log').open('x') as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=min(profile['maximum_child_seconds'],profile['deadline_unix']-time.time()-profile['closure_reserve_seconds']))
        except BaseException:
            # Retain and charge all completed or unresolved physical intents.
            pending_charge=0
            for path in (wave/'actions').glob('rank-*/*.json'):
                pending_charge+=sum(r.get('executed_or_conservative',r['requested']) for r in json.loads(path.read_text())['calls'])
            charged+=pending_charge
            write_new(wave/'FAILED.json',dict(completed=False,conservative_actions=pending_charge,
                no_automatic_retry=True,partial_outputs_preserved=True))
            raise
        result_path=wave/'exchange'/'S'/phase/'iteration-0000'/'evaluation.json'
        result=json.loads(result_path.read_text())
        if (result.get('completed') is not True or result['episodes']!=len(schedule)
                or result['physical_actions']!=maximum or result['padding_episodes']!=0
                or result['case_schedule_sha256']!=digest(schedule)):
            raise ValueError('Incomplete or altered native child; stop rather than filter it')
        charged+=maximum;episodes+=len(schedule)
        record=dict(tasks=list(tasks),phase=phase,result=result,
            evaluation=dict(path=str(result_path),sha256=sha(result_path)),
            episodes=dict(path=str(result_path.parent/'episodes.json'),sha256=sha(result_path.parent/'episodes.json')),
            profile=dict(path=str(wave/'profile.json'),sha256=sha(wave/'profile.json')))
        records.append(record)
        for task,value in result['task_results'].items():
            print(f'{phase} task={task} final_success={value["successes"]}/{value["episodes"]} actions={maximum} total_actions={charged}',flush=True)
        emit(phase+'_complete',evaluation_successes=result['task_results'])
        return record

    try:
        for batch_index,batch in enumerate(candidate_batches()):
            screened={}
            for start in range(0,len(batch),4):
                record=run_wave(batch[start:start+4],'screen')
                screened.update({int(k):v for k,v in record['result']['task_results'].items()})
            if set(screened)!=set(batch) or any(v['episodes']!=8 for v in screened.values()):
                raise ValueError('Complete fixed ranking batch required; no omitted outcomes')
            ranked=sorted((t for t in batch if 1<=screened[t]['successes']<=5),
                key=lambda t:(abs(screened[t]['successes']-2),t))
            write_new(root/f'batch-{batch_index:02d}-ranking.json',dict(tasks=batch,outcomes=screened,
                frozen_confirmation_order=ranked,ranking_before_confirmation=True))
            for task in ranked:
                if confirmations>=3:break
                record=run_wave([task],'confirm');confirmations+=1
                value=record['result']['task_results'][str(task)]
                if value['episodes']!=64:raise ValueError('Complete64 confirmation required')
                if 4<=value['successes']<=44:
                    selected=dict(task_id=task,successes=value['successes'],episodes=64,
                        confirmed_bank=record['episodes'],evaluation=record['evaluation'],
                        native_selection_frozen_before_flora=True,Flora_labels_used=0,PPO_updates=0)
                    write_new(root/'SELECTED-TASK.json',selected)
                    reason='first_native_confirmation_pass';break
            if selected is not None or confirmations>=3:
                if selected is None:reason='three_native_confirmations_failed'
                break
    except TimeoutError as error:
        reason='resource_limit';emit('stopped',stop_reason=str(error))
    except BaseException as error:
        reason='failure_preserved';emit('failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_new(root/'search-summary.json',dict(status=reason,selected=selected,confirmations=confirmations,
            physical_actions=charged,completed_episodes=episodes,children=records,
            source_commit=head,profile_sha256=sha(profile_path),model_unchanged=True,optimizer_updates=0,
            interpretation='Development selection, not an unbiased LIBERO90 score; repeated nominal placements are disclosed.'))
        emit('closed',stop_reason=reason);log.close()


if __name__=='__main__':main()
