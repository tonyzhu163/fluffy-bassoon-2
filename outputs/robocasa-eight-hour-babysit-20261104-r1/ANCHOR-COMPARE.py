"""Compare audited frozen native controls at each actual query episode."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    assert sys.platform == 'linux'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=('ID','OOD'))
    args=parser.parse_args()
    run=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
    spec=importlib.util.spec_from_file_location('qualified_reducer',Path(__file__).with_name('READOUT-COMBINE.py'))
    reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)
    rows,provenance={},{}
    for method in ('native_initial','native_final'):
        group=run/'gpu'/f'{method.replace("_","-")}-{args.split}-final-01'
        result=read(group/'RESULT.json');audit=read(group/'AUDIT-STATUS.json')
        assert result['completed'] and result['method']==method and result['split']==args.split
        assert result['cases_per_task']==50 and result['deployment_optimizer_updates']==0
        assert audit['passed'] and audit['group_result_sha256']==sha(group/'RESULT.json')
        assert sha(Path(audit['audit_summary']['path']))==audit['audit_summary']['sha256']
        provenance[method]=dict(result=dict(path=str(group/'RESULT.json'),sha256=sha(group/'RESULT.json')),
            audit_status=dict(path=str(group/'AUDIT-STATUS.json'),sha256=sha(group/'AUDIT-STATUS.json')))
        rows[method]={}
        for reference in result['source_banks']:
            path=Path(reference['path']);assert sha(path)==reference['sha256']
            bank=read(path)
            assert bank['mode']=='full' and bank['method']==method
            for episode in bank['episodes']:
                case=episode['case'];key=case['task'],case['slot'],episode['attempt']
                assert key not in rows[method]
                rows[method][key]=episode
    assert set(rows['native_initial'])==set(rows['native_final'])
    for key,initial in rows['native_initial'].items():
        final=rows['native_final'][key]
        assert initial['case']==final['case'] and initial['horizon']==final['horizon']
    tasks=('OpenDrawer','CloseToasterOvenDoor','CoffeeSetupMug') if args.split=='ID' else ('CloseFridge','TurnOnMicrowave')
    curves={}
    for task in tasks:
        curves[task]={}
        for support in range(6):
            outcomes={method:{slot:rows[method][task,slot,support+1]['success_once'] for slot in range(50)}
                for method in rows}
            seed=int.from_bytes(hashlib.sha256(f'native-anchors/{args.split}/{task}/{support}'.encode()).digest()[:4],'little')
            curves[task][str(support)]=dict(
                counts={method:sum(values.values()) for method,values in outcomes.items()},denominator=50,
                final_minus_initial_descriptive=reducer.paired(outcomes['native_initial'],outcomes['native_final'],seed))
    result=dict(completed=True,split=args.split,curves=curves,provenance=provenance,GPU_hours=0,
        meaning='Each point is success in query episode j following j-1 actual completed episodes. Both frozen native controls lack learned history conditioning; these curves are controls, not adaptation evidence. Cumulative ever-success remains separate. Paired contrasts are descriptive and do not admit source policies or choose checkpoints.')
    target=run/f'NATIVE-ANCHOR-{args.split}-RESULT.json'
    with target.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    lines=[f'# Frozen native anchor {args.split} controls','',result['meaning'],'',
        'Counts are successes out of50 in the actual query; columns denote prior completed episodes.','',
        '| Task / frozen control | 0 | 1 | 2 | 3 | 4 | 5 |','| --- | --- | --- | --- | --- | --- | --- |']
    for task in tasks:
        for method,label in (('native_initial','original SFT'),('native_final','fixed130 PPO')):
            lines.append('| '+task+' / '+label+' | '+' | '.join(str(curves[task][str(k)]['counts'][method]) for k in range(6))+' |')
    lines.extend(['','Both groups passed independent audits of every raw draw, mapped physical command, feedback, native success receipt and decoded video frame. Full paired descriptive statistics and hashed provenance are in the sibling JSON.',''])
    with target.with_suffix('.md').open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(completed=True,result=str(target),curves=curves)))


if __name__=='__main__':
    main()
