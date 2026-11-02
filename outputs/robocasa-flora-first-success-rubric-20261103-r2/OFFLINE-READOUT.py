"""Post-hoc threshold/stability diagnostics on the completed fixed panel only."""
import hashlib
import json
from pathlib import Path
import platform

assert platform.system() == 'Linux'
root = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-first-success-rubric-20261103-r2')
source = root / 'PAIRED-SAMPLE.json'
rows = json.loads(source.read_text())
assert len(rows) == 35
cases = sorted({r['episode_id'] for r in rows})
assert len(cases) == 6
results = []
for arm in ('revision1_1', 'revision1_6'):
    for threshold in (3, 4, 5):
        for required in (1, 2):
            outcomes = []
            for case in cases:
                probes = sorted((r for r in rows if r['episode_id'] == case), key=lambda r: r['control'])
                native = probes[0]['native_first_success']
                trigger, streak, previous = None, 0, None
                for probe in probes:
                    if previous is None or probe['control'] - previous != 50:
                        streak = 0
                    streak = streak + 1 if probe['labels'][arm]['score'] >= threshold else 0
                    if streak >= required:
                        trigger = probe['control']
                        break
                    previous = probe['control']
                wrong = trigger is not None and (native is None or trigger < native)
                outcomes.append(dict(id=case, native_first_success=native, first_trigger=trigger,
                    premature_or_wrong_target=wrong,
                    delay_controls=None if trigger is None or native is None else trigger-native))
            results.append(dict(arm=arm, minimum_score=threshold, required_probes=required,
                spacing_controls=50,
                false_trigger_cases=sum(r['premature_or_wrong_target'] for r in outcomes),
                successful_cases_detected=sum(r['native_first_success'] is not None and
                    r['first_trigger'] is not None and not r['premature_or_wrong_target'] for r in outcomes),
                successful_cases_missed=sum(r['native_first_success'] is not None and r['first_trigger'] is None for r in outcomes),
                outcomes=outcomes))
result = dict(post_hoc=True, fixed_development_panel=True, population_precision=False,
    consecutive_frames_tested=False, sampling_interval_seconds=2.5,
    caution='Two positive sampled prefixes do not establish uninterrupted success between them; irregular gaps reset the counter.',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), results=results)
with (root / 'OFFLINE-READOUT.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
for row in results:
    print(row['arm'], row['minimum_score'], row['required_probes'],
          'false', row['false_trigger_cases'], 'detected', row['successful_cases_detected'],
          'missed', row['successful_cases_missed'])
