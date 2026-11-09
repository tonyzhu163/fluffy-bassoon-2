"""Independently check saved replay predictions against archived expert targets."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    result = json.loads((root / 'RESULT.json').read_text())
    exit_record = json.loads((root / 'EXIT.json').read_text())
    probe = json.loads((root / 'gpu-probe.json').read_text())
    assert result['status'] == 'complete'
    assert exit_record['exit_code'] == 0
    assert probe['status'] == 'passed' and len(probe['devices']) == 1
    assert result['episodes'] == 20 and result['stride'] == 4
    assert result['frozen_parameters_unchanged'] is True
    variants = ('base', 'rgb_swap', 'unseen', 'prefix', 'wrist_swap')
    assert set(result['rows']) == set(variants)
    states = {}
    with zipfile.ZipFile(args.archive) as archive:
        for episode in range(20):
            name = f'aloha-agilex_clean_50/data/episode{episode}.hdf5'
            with h5py.File(io.BytesIO(archive.read(name)), 'r') as sample:
                states[episode] = np.array(sample['joint_action/vector'], dtype=np.float64)
    expected_meta = [(ep, t) for ep in range(20)
                     for t in range(0, len(states[ep]) - 1, 4)]
    dimensions = {'arm': [*range(6), *range(7, 13)], 'grip': [6, 13]}
    reference = {name: {} for name in (*variants, 'hold')}
    maximum_difference = 0.0
    # Predictions are serialized as float32; allow only the corresponding
    # rounding envelope when comparing metrics computed before serialization.
    with np.load(root / 'predictions.npz', allow_pickle=False) as predictions:
        assert set(predictions.files) == {'meta', *variants}
        assert predictions['meta'].tolist() == [list(x) for x in expected_meta]
        count = len(expected_meta)
        assert count == result['queries']
        for name in variants:
            assert predictions[name].shape == (count, 32, 14)
            assert predictions[name].dtype == np.float32
            assert np.isfinite(predictions[name]).all()
        for query, (episode, t) in enumerate(expected_meta):
            state = states[episode]
            targets = {shift: np.stack([state[min(t + shift + step, len(state) - 1)]
                                        for step in range(32)]) for shift in (0, 1)}
            for name in (*variants, 'hold'):
                if name == 'hold':
                    pred = np.tile(state[t], (32, 1))
                    stored = result['hold_rows'][query]
                    tolerance = 1e-12
                else:
                    raw = predictions[name][query]
                    pred = raw.astype(np.float64)
                    stored = result['rows'][name][query]
                    tolerance = float(np.max(np.spacing(np.abs(raw)))) / 2 + 1e-12
                assert (stored['episode'], stored['t']) == (episode, t)
                for shift in (0, 1):
                    for horizon in (1, 8, 32):
                        for group, dims in dimensions.items():
                            key = f's{shift}_{group}_h{horizon}'
                            value = float(np.abs(pred[:horizon, dims] -
                                                 targets[shift][:horizon, dims]).mean())
                            delta = abs(value - stored[key])
                            assert delta <= tolerance, (name, query, key, delta, tolerance)
                            maximum_difference = max(maximum_difference, delta)
                            reference[name].setdefault(key, []).append(value)
        for name in reference:
            summary = result['hold_baseline'] if name == 'hold' else result['summary'][name]
            for key, values in reference[name].items():
                for statistic, measured in [('mean', np.mean(values)),
                                            ('median', np.median(values))]:
                    assert abs(float(measured) - summary[key][statistic]) <= 1e-6
    assert len(result['hold_rows']) == count
    assert all(len(result['rows'][name]) == count for name in variants)
    report = {
        'status': 'passed', 'episodes': 20, 'queries': count,
        'policy_inferences': count * len(variants),
        'metadata_and_target_reconstruction': 'exact expected episode/stride grid',
        'maximum_row_metric_difference': maximum_difference,
        'prediction_serialization': 'float32; half-ULP envelope checked per query',
        'summary': result['summary'], 'hold_baseline': result['hold_baseline'],
        'worker_exit': exit_record,
        'files': {name: {'bytes': (root / name).stat().st_size,
                         'sha256': digest(root / name)} for name in
                  ('RESULT.json', 'predictions.npz', 'EXIT.json', 'gpu-probe.json',
                   'CPU-CHECK.json')},
        'limits': 'Teacher-forced reconstruction error is diagnostic; it does not '
                  'establish training-pipeline identity or a cause of closed-loop success differences.'
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'queries': count,
                      'policy_inferences': report['policy_inferences'],
                      'maximum_row_metric_difference': maximum_difference}))


if __name__ == '__main__':
    main()
