"""Audit actual motor records on Linux; no learned reader or model fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BOWLS = ['akita_black_bowl_1', 'akita_black_bowl_2']
OBS_SHAPES = {'agentview_image': (128, 128, 3),
              'robot0_eye_in_hand_image': (128, 128, 3),
              'robot0_eef_pos': (3,), 'robot0_eef_quat': (4,),
              'robot0_gripper_qpos': (2,)}
HORIZON = 562


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motor', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Never overwrite an audit identity'
    result = json.loads((args.motor / 'result.json').read_text())
    geometry = json.loads((args.geometry / 'result.json').read_text())
    assert result['complete'] and result['paired_initial_observations']
    assert not result['reader_trained']
    assert geometry['geometry_passed'] and geometry['balance_passed']
    assert result['public_horizon'] == HORIZON
    specs = {r['index']: r for r in geometry['layouts']}
    indices = result['indices']
    assert indices and len(set(indices)) == len(indices)
    assert len(result['results']) == 2
    observed = {}
    counterfactuals = []
    for candidate_result in result['results']:
        candidate = candidate_result['candidate']
        assert candidate in (0, 1)
        assert candidate_result['indices'] == indices
        assert len(candidate_result['records']) == len(indices)
        for index, summary in zip(indices, candidate_result['records']):
            assert summary['index'] == index and summary['candidate'] == candidate
            key = (index, candidate)
            assert key not in observed
            spec = specs[index]
            audit_path = args.motor / 'audit' / f'record-{index:02d}-{candidate}.json'
            record = json.loads(audit_path.read_text())
            for name, value in summary.items():
                assert record[name] == value, f'Summary mismatch: {key} {name}'
            assert record['hidden_rule_used_for_actions_or_termination'] is False
            assert record['initial_state_sha256'] == spec['state_sha256']
            assert record['initial_observation_hashes'] == spec['observation_hashes']
            physical = BOWLS.index(spec['candidate_object_ids'][candidate])
            assert record['selected_physical_index'] == physical
            count = record['actual_controls']
            assert 0 < count <= HORIZON
            trace = np.asarray(record['native_trace'])
            assert trace.shape == (count, 2) and np.isin(trace, [0, 1]).all()
            native_any = trace.max(axis=0).astype(int).tolist()
            assert native_any == record['native_any']
            assert record['selected_skill_success'] == native_any[physical]
            assert not trace[:-1, physical].any(), 'Selected success did not terminate raw controls'
            assert np.asarray(record['body_positions']).shape == (count, 2, 3)
            assert np.isfinite(record['body_positions']).all()
            if native_any[physical]:
                assert record['termination'] == 'selected_skill' and trace[-1, physical] == 1
            else:
                assert record['termination'] == 'fixed_horizon' and count == HORIZON
            public_path = args.motor / record['public_file']
            public_path.resolve().relative_to((args.motor / 'public').resolve())
            assert file_sha(public_path) == record['public_sha256']
            with np.load(public_path, allow_pickle=False) as public:
                assert set(public.files) == set(OBS_SHAPES) | {'actions', 'active_control_mask', 'candidate'}
                assert {k: array_sha(public[k]) for k in public.files} == record['public_array_hashes']
                assert public['candidate'].shape == () and public['candidate'].dtype == np.int64
                assert int(public['candidate']) == candidate
                mask = public['active_control_mask']
                assert mask.dtype == np.bool_ and np.array_equal(mask, np.arange(HORIZON) < count)
                actions = public['actions']
                assert actions.shape == (HORIZON, 7) and np.isfinite(actions).all()
                assert np.all(actions[count:] == 0)
                for name, shape in OBS_SHAPES.items():
                    values = public[name]
                    assert values.shape == (HORIZON + 1,) + shape
                    assert np.isfinite(values).all()
                    if name.endswith('_image'):
                        assert values.dtype == np.uint8
                    assert array_sha(values[0]) == spec['observation_hashes'][name]
                    assert np.all(values[count + 1:] == values[count]), 'Observation padding changed'
            rewards = {}
            for rule in ('leftmost', 'uppermost'):
                target_candidate = spec['rule_choices'][rule]
                target_physical = BOWLS.index(spec['candidate_object_ids'][target_candidate])
                rewards[rule] = native_any[target_physical]
            if spec['stratum'] == 'agreement':
                assert rewards['leftmost'] == rewards['uppermost']
            counterfactuals.append({'index': index, 'candidate': candidate,
                                   'public_file': record['public_file'],
                                   'shared_public_sha256_for_both_rules': record['public_sha256'],
                                   'native_outcome_by_rule': rewards})
            observed[key] = record
    assert set(observed) == {(i, c) for i in indices for c in (0, 1)}
    cells = {}
    for stratum in ('agreement', 'disagreement'):
        selected = [i for i in indices if specs[i]['stratum'] == stratum]
        distinct = []
        for index in selected:
            expected = [[int(BOWLS[j] == specs[index]['candidate_object_ids'][c])
                         for j in (0, 1)] for c in (0, 1)]
            if all(observed[index, c]['native_any'] == expected[c] for c in (0, 1)):
                distinct.append(index)
        cells[stratum] = {'paired_layouts': len(selected),
                          'candidate_successes': {str(c): sum(observed[i, c]['selected_skill_success']
                                                            for i in selected) for c in (0, 1)},
                          'distinct_one_hot_pairs': len(distinct), 'distinct_pair_indices': distinct}
    full_panel = set(indices) == set(range(16)) and len(indices) == 16
    report = {'complete': True, 'structural_record_audit_passed': True,
              'counterfactual_rule_records_share_public_bytes_by_construction': True,
              'feedback_intervention_evaluated_on_reader': False,
              'physical_episodes': len(observed), 'layout_indices': indices,
              'full_panel': full_panel, 'by_stratum': cells,
              'counterfactual_outcomes': counterfactuals,
              'motor_result_sha256': file_sha(args.motor / 'result.json'),
              'geometry_result_sha256': file_sha(args.geometry / 'result.json')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('RECORD_AUDIT', json.dumps({k: v for k, v in report.items() if k != 'counterfactual_outcomes'}), flush=True)


if __name__ == '__main__':
    main()
