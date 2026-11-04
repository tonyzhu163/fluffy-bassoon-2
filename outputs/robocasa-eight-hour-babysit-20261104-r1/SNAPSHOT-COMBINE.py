"""Reduce heterogeneous two-case deployment qualifications without final claims."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sys.platform == 'linux'
    run = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
    group = run/'gpu/direct-snapshot-ID-qualification-01'
    manifest_path = run/'inputs/direct-snapshot-01/MANIFEST.json'
    manifest = read(manifest_path)
    assert manifest['completed'] and manifest['SGD'] == 0
    methods = ('ad','dpt','dit','ic_iql')
    tasks = ('OpenDrawer','CloseToasterOvenDoor','CoffeeSetupMug')
    modes = ('full','episode')
    summaries, cases, banks, nodes, uuids = {}, {}, [], [], []
    for index, method in enumerate(methods):
        node = group/f'node-{index:02d}'
        top = read(node/'RESULT.json');binding = top['binding']
        selection = binding['selection'];frozen = manifest['methods'][method]
        assert top['completed'] and top['deployment_optimizer_updates'] == 0
        assert selection['kind'] == method and selection['qualification_only']
        assert selection['snapshot'] == frozen
        assert selection['sha256'] == frozen['candidate']['sha256']
        assert sha(Path(frozen['candidate']['path'])) == selection['sha256']
        assert binding['stage'] == 'qualification' and binding['split'] == 'ID'
        assert binding['cases_per_task'] == 2 and binding['support_episodes'] == list(range(6))
        assert binding['memory_modes'] == list(modes)
        assert binding['shard_count'] == 1 and binding['shard_index'] == 0
        assert binding['signature']['all_parameters_frozen']
        uuids.append(binding['signature']['scheduler_uuid'])
        nodes.append(dict(path=str(node/'RESULT.json'),sha256=sha(node/'RESULT.json')))
        summaries[method] = {}
        found = sorted(node.glob('*/*/bank-*/RESULT.json'))
        assert len(found) == 6
        for bank in found:
            row = read(bank);mode, task = row['mode'],row['task']
            assert row['method'] == method and mode in modes and task in tasks
            assert row['denominators'] == 2 and len(row['episodes']) == len(row['references']) == 12
            assert [item['case']['slot'] for item in row['cases']] == [0,1]
            assert row['actual_batch_sizes'] == [2]*72 and row['controls_charged'] == 7200
            for reference in row['references']:
                assert sha(Path(reference['path'])) == reference['sha256']
            for episode in row['episodes']:
                case = episode['case'];slot=case['slot'];attempt=episode['attempt']
                key = method,mode,task,slot,attempt
                assert key not in cases and slot in (0,1) and 1 <= attempt <= 6
                assert episode['completed'] and episode['horizon'] == 600
                assert case['bank'] == 'joint_meta_qualification_ID' and case['task'] == task
                assert case['native_seed'] == 63105000 + tasks.index(task)*10000 + slot
                cases[key] = episode
            counts = {str(support):sum(ep['success_once'] for ep in row['episodes'] if ep['attempt']==support+1)
                for support in range(6)}
            assert counts == row['first_success_counts_by_support_episodes']
            summaries[method].setdefault(mode,{})[task] = dict(cases=2,first_success_by_support_episodes=counts)
            banks.append(dict(path=str(bank),sha256=sha(bank)))
    assert len(set(uuids)) == 4
    assert set(cases) == {(method,mode,task,slot,attempt) for method in methods for mode in modes
        for task in tasks for slot in range(2) for attempt in range(1,7)}
    result = dict(completed=True,qualification_only=True,training_continues=True,
        cases_per_task=2,episodes=len(cases),physical_controls=len(cases)*600,
        source_nodes=nodes,source_banks=banks,summaries=summaries,
        frozen_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
        selected_candidates={method:manifest['methods'][method]['candidate'] for method in methods},
        deployment_optimizer_updates=0,
        meaning='Actual six-episode native deployment/history qualification only; no convergence, final competence, method ranking or adaptation claim.',
        native_truth_audit_status=str(group/'AUDIT-STATUS.json'))
    target=group/'RESULT.json'
    with target.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_nodes','source_banks')}))


if __name__ == '__main__':
    main()
