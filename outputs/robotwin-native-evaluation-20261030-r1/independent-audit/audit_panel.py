"""Read-only artifact audit. Standard library only; no repo/model imports."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ast
import hashlib
import json
import struct
import zipfile

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-evaluation-20261030-r1')
EVAL=ROOT/'evaluation'
CODE=Path('/mowenjie/fluffy-bassoon-2/worktrees/robotwin-history-live-wrapper-20261030-r2')
PIN='2fd2096906e52aabeaeb984f333fe1aaa6c4e196'
PLAN_SHA='b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c'
CANONICAL='3e82a6df6001024cffaae92af26a2412fb11b260302504ec1ae4243d06b6b5f1'
BACKEND='eacc64b4daf8f78aeb5e023a2c4d2a2a040eacfd2e662335c118334c2ab50ff3'
errors=[]
checks=0


def check(ok,message):
    global checks
    checks+=1
    if not ok: errors.append(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def mask_count(path):
    """Inspect tiny stored bool-mask data/NPY header, without NumPy imports."""
    with zipfile.ZipFile(path) as archive:
        check(archive.testzip() is None,'native NPZ CRC: '+str(path))
        value=archive.read('valid_action_mask.npy')
    check(value[:6]==b'\x93NUMPY','mask NPY magic: '+str(path))
    version=value[6]
    size=2 if version==1 else 4
    length=int.from_bytes(value[8:8+size],'little')
    metadata=ast.literal_eval(value[8+size:8+size+length].decode())
    body=value[8+size+length:]
    check(metadata['descr'] in ('|b1','|?') and not metadata['fortran_order'], 'mask dtype/layout: '+str(path))
    check(set(body)<={0,1},'mask nonboolean data: '+str(path))
    shape=metadata['shape'];elements=1
    for dimension in shape:elements*=dimension
    check(elements==len(body),'mask size: '+str(path))
    return sum(body)


plan_path=CODE/'outputs/robotwin-native-evaluation-plan-20261030-r1/plan.json'
plan=read(plan_path);summary=read(EVAL/'summary.json');progress=read(EVAL/'progress.json')
check(summary==progress,'summary/progress differ')
check(summary['status']=='complete' and summary['completed_unit_count']==summary['requested_unit_count']==18,'panel not complete')
check((ROOT/'exit-code.txt').read_text().strip()=='0','wrapper exit nonzero')
check(summary['source_commit']==PIN and summary['plan_sha256']==PLAN_SHA,'summary pin/plan')
check(sha(plan_path)==PLAN_SHA,'plan file SHA')
check(hashlib.sha256(canonical(plan).encode()).hexdigest()==CANONICAL,'canonical plan SHA')
check(sha(CODE/'src/fluffy_bassoon/rl/robotwin_noise.py')==BACKEND,'native backend bytes')
check(plan['native_backend_sha256']==BACKEND,'plan backend identity')
check(plan['episodes_per_trial']==8 and plan['episode_limit']==400 and plan['max_actions']==57600,'panel dimensions')
check(plan['reported_episode_indices']==[1,2,4,8] and plan['additional_deployment_sgd_updates']==0,'reporting/SGD contract')
check(summary['additional_sgd_updates']==0 and summary['error'] is None,'summary SGD/error')
probe=read(EVAL/'gpu-probe.json');admission=read(EVAL/'admission.json');execution=read(EVAL/'execution.json')
check(probe['status']=='passed' and len(probe['devices'])==1,'GPU probe not passed single GPU')
pci=probe['devices'][0]['pci_bus_id']
check(admission['plan_sha256']==CANONICAL and admission['model_endpoint']==100,'admission endpoint/plan')
check(admission['additional_robot_actions']==0 and len(admission['checked'])==5,'CPU admission conditions')
parameter_shas={row['condition']:row['parameter_sha256'] for row in admission['checked']}
model_files={key:dict(path=ref['path'],sha256=sha(ref['path']),planned_sha256=ref['sha256']) for key,ref in plan['models'].items()}
model_files['sac5-mean']=dict(path=plan['teacher']['path'],sha256=sha(plan['teacher']['path']),planned_sha256=plan['teacher']['sha256'])
for name,item in model_files.items():check(item['sha256']==item['planned_sha256'],'model file SHA: '+name)
check(sha(plan['firstfit_summary']['path'])==plan['firstfit_summary']['sha256'],'first-fit summary SHA')
check(read(plan['firstfit_summary']['path'])['status']=='completed_all_four','first-fit completed bank')
check(sha(plan['capture_qualification']['path'])==plan['capture_qualification']['sha256'],'capture qualification SHA')
capture=read(plan['capture_qualification']['path'])
check(capture.get('passed') is True and capture.get('status')=='passed','capture qualification status')

# Reconstruct the immutable physical journal without acquiring its lock.
ledger_root=ROOT/'evaluation-ledger';snapshot=read(ledger_root/'snapshot.json')
files=sorted((ledger_root/'events').glob('*.json'))
with ThreadPoolExecutor(max_workers=16) as pool:events=list(pool.map(read,files))
decisions={};runs={};receipts={};previous=None;header=None;prefixes={}
for sequence,(path,event) in enumerate(zip(files,events)):
    event=dict(event);recorded=event.pop('sha256')
    actual=hashlib.sha256(canonical(event).encode()).hexdigest()
    check(path.name==f'{sequence:010d}.json' and event['sequence']==sequence and event['previous_sha256']==previous and recorded==actual,'journal event '+str(sequence))
    previous=actual;kind=event['kind']
    if kind=='header':header=event['header']
    elif kind=='run':runs[event['run_id']]=event['source_commit']
    elif kind=='reserve':decisions[event['decision_id']]=dict(event['request'],granted=event['granted'],status='pending',actual_actions=None)
    elif kind=='resolve':decisions[event['decision_id']].update(status='resolved',actual_actions=event['actual_actions'])
    elif kind=='uncertain':decisions[event['decision_id']]['status']='uncertain'
    elif kind=='receipt':receipts[event['import_id']]=event['receipt']
    else:check(False,'unknown event '+kind)
    prefixes[actual]=sequence+1
counts=dict(charged_actions=0,actual_actions=0,pending_actions=0,uncertain_actions=0,by_phase={},by_method={})
unresolved={}
for identity,item in list(decisions.items())+list(receipts.items()):
    status=item.get('status','resolved');cost=item['actual_actions'] if status=='resolved' else item['granted']
    counts['charged_actions']+=cost;counts['actual_actions' if status=='resolved' else status+'_actions']+=cost
    phase=item['phase'];method=item['method'];key=phase+'/'+method
    counts['by_phase'][phase]=counts['by_phase'].get(phase,0)+cost
    counts['by_method'][key]=counts['by_method'].get(key,0)+cost
    if status!='resolved':unresolved[identity]=item
rebuilt=dict(header=header,receipts=receipts,runs=runs,**counts,decision_count=len(decisions),unresolved_decisions=unresolved,journal_events=len(files),journal_sha256=previous)
check(rebuilt==snapshot==summary['budget'],'journal/snapshot/summary equality')
check(header['total_actions']==57600 and header['phase_caps']=={'target':57600} and header['source_caps']=={},'separate target-only action ceiling')
check(counts['by_phase']=={'target':counts['actual_actions']} and counts['pending_actions']==counts['uncertain_actions']==0 and not unresolved,'closed target-only ledger')
check(all(pin==PIN for pin in runs.values()),'ledger run pins')

units=[];native_banks={};ledger_ids_seen=set();previous_cumulative=0
expected_order=[(condition,task) for condition in plan['conditions'] for task in plan['cases']]
for index,(ref,expected) in enumerate(zip(summary['completed_units'],expected_order)):
    label=f'{index:02d} {expected[0]} {expected[1]}'
    condition,task=expected;unit=Path(ref['receipt_path']).parent
    check((ref['condition'],ref['task'])==expected,'unit order '+label)
    check(sha(ref['receipt_path'])==ref['receipt_sha256'],'unit receipt SHA '+label)
    receipt=read(ref['receipt_path']);result_path=unit/'trial/result.json';result=read(result_path)
    check(sha(result_path)==receipt['result_sha256'],'result SHA '+label)
    check(receipt['completed'] and receipt['condition']==condition and receipt['task']==task and receipt['source_commit']==PIN and receipt['backend_sha256']==BACKEND and receipt['plan_sha256']==PLAN_SHA,'unit identity '+label)
    check(result['completed'] and result['task']==task and result['context']['method']==condition and result['context']['phase']=='target' and result['context']['evaluation_plan_sha256']==CANONICAL,'result identity '+label)
    process=read(unit/'process.json');intent=read(unit/'intent.json');trial_intent=read(unit/'trial/intent.json')
    check(process['returncode']==0 and not process['timed_out'],'child exit '+label)
    check(intent['source_commit']==PIN and intent['plan_sha256']==PLAN_SHA and intent['unit_index']==index,'child intent '+label)
    check(result['context']==trial_intent['context'] and trial_intent['seeds']==plan['cases'][task],'trial intent '+label)
    check(len(result['episodes'])==len(result['reset_cases'])==8,'K8 count '+label)
    check(result['reported_episodes']=={str(k):result['episodes'][k-1]['success'] for k in (1,2,4,8)},'reported episodes '+label)
    check(result['support_episodes']==[1,2,3,4] and result['query_episodes']==[5,6,7,8],'episode split '+label)
    if condition=='native':native_banks[task]=result['reset_cases']
    check(result['reset_cases']==native_banks[task],'native bank reset identity '+label)
    for episode,(case,row) in enumerate(zip(result['reset_cases'],result['episodes']),1):
        check(case['episode']==episode and case['task']==task and case['requested_seed']==plan['cases'][task][episode-1] and type(case['actual_seed']) is int,'fixed episode identity '+label)
        check(read(unit/f'trial/reset-{episode:02d}.json')==case and read(unit/f'trial/episode-{episode:02d}.json')==row,'episode files '+label)
        check(all(row[key]==value for key,value in case.items()),'episode reset fields '+label)
        check(0<=row['primitive_steps']<=400,'episode action duration '+label)
    context=result['context'];base=condition.removesuffix('-reset')
    check(context['offline_weights_frozen'] is True and result['offline_weights_unchanged'] is True,'frozen steering '+label)
    check(context['erase_each_episode']==condition.endswith('-reset'),'reset ablation '+label)
    if condition not in ('native','native-zero'):
        identity=model_files[base]
        check(context['policy_sha256']==parameter_shas[base] and context['policy_checkpoint']==identity['path'] and context['policy_checkpoint_sha256']==context['policy_version']==identity['sha256'],'model/admission identity '+label)
    qualification=read(unit/'qualification.json')
    check(qualification['parity']==receipt['parity'] and qualification['parity']['passed'] and qualification['parity']['max_abs_error']==0.0,'native noise parity '+label)
    check(qualification['capture']['frozen_fingerprint']==plan['frozen_motor_sha256'] and qualification['capture']['sha256']==plan['capture_qualification']['sha256'],'motor qualification identity '+label)
    check(qualification['zero_physical_actions'] and qualification['preflight_reset']['reset_policy_actions']==0,'zero-action qualification '+label)
    renderers=receipt['runtime']['renderer_records']
    for path,expected_sha in renderers.items():
        check(sha(path)==expected_sha,'renderer receipt SHA '+path)
        renderer=read(path);check(renderer['actual_render_pci']==renderer['expected_cuda_pci']==pci,'allocated renderer '+label)
    check(all(path in renderers and renderers[path]==value for path,value in result['renderer_records'].items()),'result renderer subset '+label)
    runtime=read(unit/'corpus/runtime-contract.json')
    check(runtime['actual_render_pci']==pci and runtime['norm_stats_sha256']==plan['runtime_contract']['norm_stats_sha256'],'runtime PCI/norm '+label)
    corpus=unit/'corpus';stats=read(corpus/'statistics.json')
    check(stats==receipt['corpus'],'corpus receipt statistics '+label)
    check(not stats['unresolved_decisions'] and stats['native_completed_chunks']==0 and stats['pending_decision'] is None,'closed raw corpus '+label)
    corpus_actions=corpus_chunks=0;native_seconds=0.;target_episodes=0
    for reset_path in (corpus/'episodes').glob('*/reset.json'):
        reset=read(reset_path);manifest=read(reset_path.parent/'manifest.json');phase=reset['context']['phase']
        check(not manifest.get('pending_decision'),'raw pending '+label)
        if phase=='target':
            target_episodes+=1
            check(reset['reset_info']['reset_policy_actions']==0,'raw reset action '+label)
            matching=[case for case in result['reset_cases'] if case['requested_seed']==reset['reset_info']['requested_seed']]
            check(len(matching)==1 and all(reset['reset_info'][key]==matching[0][key] for key in ('task','requested_seed','actual_seed','observation_sha256','features_sha256')),'raw reset/native bank '+label)
        else:check(phase=='preflight' and not manifest['chunks'],'non-target raw actions '+label)
        actions=0
        for chunk_ref in manifest['chunks']:
            path=reset_path.parent/chunk_ref['path'];check(sha(path)==chunk_ref['sha256'],'chunk JSON SHA '+label)
            row=read(path);check(row['status']=='complete' and row['context']['phase']=='target' and row['context']['method']==condition,'raw target decision '+label)
            identity=row['context']['campaign_decision_id'];check(identity not in ledger_ids_seen,'duplicate raw decision '+identity);ledger_ids_seen.add(identity)
            decision=decisions.get(identity);actual=row['native_result']['executed_primitive_steps']
            check(decision is not None and decision['status']=='resolved' and decision['actual_actions']==actual and decision['method']==condition and decision['task']==task,'raw/ledger action '+identity)
            array_ref=row['native_result']['arrays'];array_path=corpus/array_ref['path']
            check(sha(array_path)==array_ref['sha256'],'native result array SHA '+label)
            check(mask_count(array_path)==actual,'native valid-action mask duration '+label)
            actions+=actual;native_seconds+=row['native_result']['duration_seconds'];corpus_chunks+=1
        check(actions==manifest['executed_primitive_steps'],'episode manifest duration '+label)
        corpus_actions+=actions
    check(target_episodes==8,'raw target episodes '+label)
    check(corpus_actions==result['actual_actions']==stats['executed_primitive_steps']==sum(row['primitive_steps'] for row in result['episodes']),'closed corpus/result actions '+label)
    check(corpus_chunks==stats['completed_chunks']==sum(row['chunks'] for row in result['episodes'])==result['motor_queries'],'decision totals '+label)
    cumulative=receipt['budget']['actual_actions']
    check(receipt['budget']['pending_actions']==receipt['budget']['uncertain_actions']==0 and not receipt['budget']['unresolved_decisions'],'closed unit ledger '+label)
    check(cumulative-previous_cumulative==corpus_actions and receipt['budget']['journal_sha256'] in prefixes and prefixes[receipt['budget']['journal_sha256']]==receipt['budget']['journal_events'],'per-unit ledger boundary '+label)
    previous_cumulative=cumulative
    units.append(dict(condition=condition,task=task,receipt_sha256=ref['receipt_sha256'],result_sha256=receipt['result_sha256'],
        successes=sum(row['success'] for row in result['episodes']),success_vector=[int(row['success']) for row in result['episodes']],
        reset_cases=result['reset_cases'],episodes=result['episodes'],reported_episodes=result['reported_episodes'],
        actual_actions=corpus_actions,chunks=corpus_chunks,native_step_seconds=native_seconds,
        elapsed_seconds=result['elapsed_seconds'],child_elapsed_seconds=process['elapsed_seconds'],
        parameter_sha256=context['policy_sha256'],checkpoint_sha256=context['policy_checkpoint_sha256'],
        renderer_receipts=len(renderers),parity_max_abs_error=receipt['parity']['max_abs_error']))
check(len(units)==18 and set(ledger_ids_seen)==set(decisions),'all physical decisions represented in corpus')
check(sum(unit['actual_actions'] for unit in units)==snapshot['actual_actions'],'full panel physical action sum')
comparisons=[]
lookup={(unit['condition'],unit['task']):unit for unit in units}
for task in plan['cases']:
    for family in ('ad','dpt'):
        full=lookup[(family+'-full',task)]
        for other_name in (family+'-static',family+'-full-reset'):
            other=lookup[(other_name,task)];changes=[]
            for a,b in zip(full['episodes'],other['episodes']):
                if a['success']!=b['success']:
                    changes.append(dict(episode=a['episode'],requested_seed=a['requested_seed'],actual_seed=a['actual_seed'],
                        observation_sha256=a['observation_sha256'],features_sha256=a['features_sha256'],full_success=a['success'],other_success=b['success']))
            comparisons.append(dict(task=task,full=family+'-full',other=other_name,full_successes=full['successes'],other_successes=other['successes'],changes=changes))
report=dict(status='passed' if not errors else 'failed',checks=checks,errors=errors,
    source_commit=PIN,backend_sha256=BACKEND,plan_file_sha256=PLAN_SHA,canonical_plan_sha256=CANONICAL,
    wrapper_status=summary['status'],wrapper_exit_code=int((ROOT/'exit-code.txt').read_text()),
    scheduler_status='Not independently queried; wrapper completion/exit0 is separately verified.',
    elapsed_seconds=summary['elapsed_seconds'],cumulative_gpu_seconds_lower_bound=summary['cumulative_gpu_seconds_lower_bound'],
    total_actual_actions=snapshot['actual_actions'],total_actions_cap=57600,remaining_actions=57600-snapshot['charged_actions'],
    decisions=len(decisions),journal_events=len(files),journal_sha256=previous,journal_snapshot_equal=rebuilt==snapshot,
    additional_sgd_updates=0,model_endpoint=admission['model_endpoint'],model_files=model_files,parameter_admission=parameter_shas,
    frozen_motor_sha256=plan['frozen_motor_sha256'],allocated_render_pci=pci,native_banks=native_banks,
    units=units,paired_comparisons=comparisons,
    limits=['No model constructors/forwards/tests/CUDA or ledger constructors were invoked by this audit.',
            'No historical RGB frame or observation NPZ was rehashed. Pairing uses hash-bound reset metadata and native-bank equality.',
            'Model parameter digests were matched to CPU admission and frozen-result contexts, not recomputed from model tensors.',
            'Pinned run_trial checks steering parameter equality and recomputes the motor fingerprint before publishing each complete result; separate literal motor-after fields are not retained.',
            'One development K8 trial per target/condition, fixed order and single fit endpoint; no ICRL or statistical generalization claim.'])
print(json.dumps(report,indent=2,allow_nan=False))
