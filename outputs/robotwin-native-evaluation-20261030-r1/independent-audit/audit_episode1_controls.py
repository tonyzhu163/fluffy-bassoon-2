"""Read-only equality of saved E1 features/noise/actions; no model imports."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-evaluation-20261030-r1/evaluation')


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(value).hexdigest()


def member(root,ref,name):
    with zipfile.ZipFile(root/ref['path']) as archive:
        return archive.read(name+'.npy')


def episode1(condition,task):
    summary=read(ROOT/'summary.json')
    ref=next(row for row in summary['completed_units'] if row['condition']==condition and row['task']==task)
    unit=Path(ref['receipt_path']).parent;corpus=unit/'corpus';result=read(unit/'trial/result.json')
    target=result['reset_cases'][0]
    matches=[]
    for path in (corpus/'episodes').glob('*/reset.json'):
        reset=read(path)
        if reset['context']['phase']=='target' and reset['reset_info']['requested_seed']==target['requested_seed']:
            matches.append(path)
    if len(matches)!=1:raise ValueError('E1 reset identity is not unique')
    reset_path=matches[0];manifest=read(reset_path.parent/'manifest.json')
    rows=[];arrays=[]
    for ref in manifest['chunks']:
        path=reset_path.parent/ref['path'];row=read(path)
        observation=member(corpus,row['observation'],'features')
        following=member(corpus,row['next_observation'],'features')
        noise=member(corpus,row['decision_arrays'],'sampled_noise')
        motor=member(corpus,row['decision_arrays'],'issued_motor_joint_targets')
        arrays.append(dict(features=observation,next_features=following,noise=noise,motor=motor))
        rows.append(dict(chunk=row['chunk_index'],observation_features_npy_sha256=digest(observation),
            next_features_npy_sha256=digest(following),noise_npy_sha256=digest(noise),motor_targets_npy_sha256=digest(motor),
            sampled_noise_recorded_array_sha256=row['decision_arrays']['arrays']['sampled_noise']['sha256'],
            issued_motor_recorded_array_sha256=row['decision_arrays']['arrays']['issued_motor_joint_targets']['sha256'],
            reward=row['learning_result']['reward'],terminated=row['learning_result']['terminated'],
            truncated=row['learning_result']['truncated'],primitive_steps=row['native_result']['executed_primitive_steps']))
    return dict(condition=condition,task=task,episode=result['episodes'][0],reset=target,
                context=result['context'],trace=rows),arrays


report=[]
for family in ('ad','dpt'):
    for task in ('adjust_bottle','move_stapler_pad'):
        full,a=episode1(family+'-full',task);reset,b=episode1(family+'-full-reset',task)
        comparisons=[]
        for index,(left,right) in enumerate(zip(a,b),1):
            same={key:left[key]==right[key] for key in left}
            x=full['trace'][index-1];y=reset['trace'][index-1]
            same['reward_end_duration']=all(x[key]==y[key] for key in ('reward','terminated','truncated','primitive_steps'))
            comparisons.append(dict(chunk=index,equal=same))
        first_divergence={field:next((row['chunk'] for row in comparisons if not row['equal'][field]),None)
                          for field in ('features','next_features','noise','motor','reward_end_duration')}
        report.append(dict(family=family,task=task,full=full,reset=reset,
            identical_weights=full['context']['policy_sha256']==reset['context']['policy_sha256']
                and full['context']['policy_checkpoint_sha256']==reset['context']['policy_checkpoint_sha256'],
            identical_reset=full['reset']==reset['reset'],first_decision_equal=comparisons[0]['equal'],
            first_divergence=first_divergence,comparison=comparisons,
            complete_episode_trace_equal=len(a)==len(b) and all(all(row['equal'].values()) for row in comparisons)))
print(json.dumps(dict(comparisons=report,limits='Saved NPY member bytes only: features/noise/issued motor targets. RGB members were not read. Initial model token is not separately stored; matching feature bytes and first-noise/action are directly inspected.'),indent=2))
