"""Verify the two complete AD arms and every audited, staged cold wave."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
tasks = dict(OpenDrawer=600, CloseToasterOvenDoor=600, CoffeeSetupMug=600,
             CloseFridge=900, TurnOnMicrowave=450)
candidates = dict(kv_bridge='f7085e4a6940546e32e46710314ac4fa4319e6179e69e23320e486d48345ad74',
                  velocity_bridge='04701478c2f3f850b8ce0329855f334732212d4186ae9502e4d97a08a3bab9ca')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


admission = read(args.run / 'READOUT-ADMISSION.json')
assert admission['approved_GPU_hours'] == 150
assert set(admission['admitted_model_keys']) == {'kv_bridge/ad', 'velocity_bridge/ad'}
held = {f'{i}/{m}' for i in candidates for m in ('dpt', 'dit', 'ic_iql')}
assert set(admission['not_admitted_model_keys']) == held
assert all(not (args.run / 'readouts' / key).exists() for key in held)
models, references = {}, []
for interface, candidate in candidates.items():
    total = dict(episodes=0, physical_controls=0, native_chunks=0, decoded_videos=0,
                 video_frames=0, banks=0, cold_payload_bytes=0, cold_archive_bytes=0)
    for split, names in [('ID', list(tasks)[:3]), ('OOD', list(tasks)[3:])]:
        group = args.run / 'readouts' / interface / 'ad' / split
        result, summary, binding = (read(group / n) for n in ('RESULT.json', 'SUMMARY.json', 'BINDING.json'))
        assert result['completed'] and summary['completed']
        assert result['deployment_optimizer_updates'] == summary['deployment_optimizer_updates'] == 0
        assert binding['source_commit'] == '87984718c9199797d31a158c47225cc695623afe'
        assert binding['runtime_commit'] == '8f9bfa7661928ef8c17494b04f4d7bf3125634e1'
        assert binding['model_sha256'] == '661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
        assert binding['norm_sha256'] == '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
        assert binding['selected_candidate_sha256'] == summary['selected_candidate_sha256'] == candidate
        assert sha(group / 'selected-candidate.pt') == candidate
        binding_sha = sha(group / 'BINDING.json')
        for task in names:
            modes = {}
            for mode in ('full', 'episode'):
                cases = {}
                for first in range(0, 50, 8):
                    bank = group / mode / task / f'bank-{first:03d}'
                    raw, audit, marker = (read(bank / n) for n in ('RESULT.json', 'AUDIT.json', 'PACKED-ARCHIVE.json'))
                    manifest = Path(marker['manifest'])
                    packed = read(manifest)
                    eviction = read(manifest.parent.parent / 'RECEIPT.json')
                    size = min(8, 50 - first)
                    assert raw['binding_sha256'] == binding_sha
                    assert raw['interface'] == interface and raw['method'] == 'ad'
                    assert raw['mode'] == mode and raw['task'] == task
                    assert raw['optimizer_updates'] == audit['SGD'] == 0 and raw['native_parameters_frozen']
                    assert raw['denominators'] == size and len(raw['episodes']) == size * 6
                    assert raw['actual_batch_sizes'] == [size] * (6 * tasks[task] // 50)
                    expected_slots = list(range(first, first + size))
                    assert [c['case']['slot'] for c in raw['cases']] == expected_slots
                    for case in raw['cases']:
                        assert case['horizon'] == tasks[task]
                        assert case['case']['native_seed'] == 64105000 + list(tasks).index(task) * 10000 + case['case']['slot']
                    assert audit['passed'] and audit['all_commands_and_draws_reproduced']
                    assert audit['all_initial_physical_public_states_match']
                    assert audit['memory_mode'] == mode
                    assert audit['all_prior_history_retained'] == (mode == 'full')
                    assert audit['episodes'] == size * 6
                    assert audit['physical_controls'] == raw['controls_charged'] == size * 6 * tasks[task]
                    assert audit['complete_chunks'] == size * 6 * tasks[task] // 50
                    assert audit['decoded_videos'] == size * 6 * 3
                    assert audit['video_frames'] == size * 6 * (tasks[task] + 1) * 3
                    digest = sha(bank / 'RESULT.json')
                    assert audit['result_sha256'] == marker['result_sha256'] == digest
                    assert marker['verified'] and marker['original_metadata_paths_and_bytes_preserved']
                    assert packed['passed'] and packed['all_stageback_bytes_verified']
                    assert packed['independent_source_audit_sha256'] == sha(bank / 'AUDIT.json')
                    assert packed['archive_sha256'] == marker['archive_sha256']
                    assert packed['cold'] == marker['cold'] == eviction['cold']
                    assert eviction['archive_sha256'] == packed['archive_sha256']
                    assert marker['raw_stageback_required'] and eviction['raw_stageback_required']
                    assert Path(packed['cold']).stat().st_size == packed['archive_bytes']
                    assert packed['payload_bytes'] == sum(v['bytes'] for v in packed['files'])
                    assert sum(v['relative'].endswith('.mp4') for v in packed['files']) == audit['decoded_videos']
                    assert eviction['completed'] and eviction['all_raw_bytes_preserved_and_stageback_SHA_verified']
                    for episode in raw['episodes']:
                        assert episode['completed'] and episode['horizon'] == tasks[task]
                        key = json.dumps(episode['case'], sort_keys=True)
                        item = cases.setdefault(key, {})
                        assert episode['attempt'] not in item
                        item[episode['attempt']] = (int(episode['success_once']), int(episode['success_at_end']))
                    for name, value in [('episodes', audit['episodes']), ('physical_controls', audit['physical_controls']),
                                        ('native_chunks', audit['complete_chunks']), ('decoded_videos', audit['decoded_videos']),
                                        ('video_frames', audit['video_frames']), ('banks', 1),
                                        ('cold_payload_bytes', packed['payload_bytes']), ('cold_archive_bytes', packed['archive_bytes'])]:
                        total[name] += value
                    references.append(dict(bank=str(bank), result_sha256=digest, audit_sha256=sha(bank / 'AUDIT.json'),
                                           manifest_sha256=sha(manifest), archive_sha256=packed['archive_sha256']))
                assert len(cases) == 50 and all(set(v) == set(range(1, 7)) for v in cases.values())
                modes[mode] = cases
                row = summary['curves'][task][mode]
                assert row['denominator'] == 50
                assert row['first_success'] == [sum(v[k][0] for v in cases.values()) for k in range(1, 7)]
                assert row['final_success'] == [sum(v[k][1] for v in cases.values()) for k in range(1, 7)]
                assert row['ever_success'] == [sum(max(v[j][0] for j in range(1, k + 1)) for v in cases.values()) for k in range(1, 7)]
            assert modes['full'].keys() == modes['episode'].keys()
            for k in range(1, 7):
                deltas = [modes['full'][c][k][0] - modes['episode'][c][k][0] for c in modes['full']]
                row = summary['paired_full_minus_reset'][task][str(k - 1)]
                assert math.isclose(row['delta_percentage_points'], sum(deltas) * 2, abs_tol=1e-10)
                assert row['wins'] == sum(d > 0 for d in deltas) and row['losses'] == sum(d < 0 for d in deltas)
    assert (total['episodes'], total['physical_controls'], total['banks'], total['native_chunks'],
            total['decoded_videos'], total['video_frames']) == (3000, 1890000, 70, 37800, 9000, 5679000)
    models[interface] = total
report = dict(passed=True, stage='Balanced AD interface pair only; full eight-model matrix remains incomplete',
              checked_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              model_totals=models, episodes=6000, physical_controls=3780000, banks=140,
              decoded_videos=18000, video_frames=11358000, deployment_optimizer_updates=0,
              scientific_code_commit='87984718c9199797d31a158c47225cc695623afe',
              candidates=candidates, all_complete_raw_audits_and_cold_full_readback_member_stageback_receipts_verified=True,
              raw_reader_requires_explicit_stageback=True, all_six_other_models_unadmitted=True, source_banks=references)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'source_banks'}))
