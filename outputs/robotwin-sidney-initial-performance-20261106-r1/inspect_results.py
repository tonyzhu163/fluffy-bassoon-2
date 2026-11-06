"""Read completed artifacts; no model, simulator, tensor library or new episodes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def npz_members(path):
    # Hash saved array records without loading or computing model tensors.
    with zipfile.ZipFile(path) as archive:
        return {n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args(); root = args.output
    result_sha = sha(root/'RESULT.json'); exit_sha = sha(root/'EXIT.json')
    report = json.loads((root/'RESULT.json').read_text())
    exit_record = json.loads((root/'EXIT.json').read_text())
    assert report['status'] == 'complete' and exit_record['exit_code'] == 0
    assert report['frozen_parameter_versions_unchanged'] and report['optimizer_updates'] == 0
    assert report['completed_episodes'] == len(report['episodes']) == 80
    assert len(report['cases']) == 16 and len(report['modes']) == 5
    episodes = {(e['case_index'],e['mode']):e for e in report['episodes']}
    assert len(episodes) == 80 and all(e['complete'] for e in episodes.values())
    summaries = {}; native_receipts = 0
    for ci, seed in enumerate(report['cases']):
        initial = None
        for mode in report['modes']:
            e = episodes[ci,mode]; directory = root/f'case-{ci:02d}'/mode
            assert e == json.loads((directory/'episode.json').read_text())
            assert e['initial'] == json.loads((directory/'initial.json').read_text())
            assert e['seed'] == seed and e['case_index'] == ci
            if initial is None: initial = e['initial']
            assert e['initial'] == initial
            assert e['success'] == any(c['success'] for c in e['chunks'])
            assert e['requested_controls'] == len(e['chunks'])*50 <= 400
            assert (directory/'initial.jpg').is_file() and (directory/'final.jpg').is_file()
            if mode == 'sde5_fast': assert e['measured_executed_controls'] is None
            else:
                assert e['measured_executed_controls'] == sum(c['measured_executed_controls'] for c in e['chunks'])
                assert all(0<c['measured_executed_controls']<=50 for c in e['chunks'])
            for c in e['chunks']:
                assert c['noise_seed'] == 1000+ci*100+c['chunk']
                obs = directory/f"chunk-{c['chunk']:02d}-observation.npz"; assert obs.is_file()
                ext = 'npz' if mode == 'native10' else 'npy'
                action = directory/f"chunk-{c['chunk']:02d}-actions.{ext}"; assert action.is_file()
                if mode == 'native10':
                    receipt = json.loads(action.with_suffix('.json').read_text())
                    assert receipt['frozen_unchanged']; native_receipts += 1
    for mode in report['modes']:
        es = [e for (_,m),e in episodes.items() if m == mode]
        s = dict(episodes=len(es),successes=sum(e['success'] for e in es))
        assert s == report['summaries'][mode]
        summaries[mode] = dict(s,success_percent=100*s['successes']/len(es),
            requested_controls=sum(e['requested_controls'] for e in es),
            measured_executed_action_boundaries=(None if mode=='sde5_fast' else sum(e['measured_executed_controls'] for e in es)),
            chunks=sum(len(e['chunks']) for e in es),
            inference_wall_seconds=sum(c['inference_seconds'] for e in es for c in e['chunks']))
    requested = sum(e['requested_controls'] for e in episodes.values())
    assert requested == report['requested_controls_total'] <= 32000
    fast_full = []
    for ci in range(16):
        a,b = episodes[ci,'sde5_full'],episodes[ci,'sde5_fast']
        checks = dict(success=a['success']==b['success'],chunk_count=len(a['chunks'])==len(b['chunks']))
        chunks = []
        for ca,cb in zip(a['chunks'],b['chunks']):
            da = root/f'case-{ci:02d}'/'sde5_full'; db = root/f'case-{ci:02d}'/'sde5_fast'
            index = ca['chunk']; assert index == cb['chunk']
            chunks.append(dict(chunk=index,
                actions_npy_exact=sha(da/f'chunk-{index:02d}-actions.npy')==sha(db/f'chunk-{index:02d}-actions.npy'),
                observation_members_exact=npz_members(da/f'chunk-{index:02d}-observation.npz')==npz_members(db/f'chunk-{index:02d}-observation.npz'),
                output_metadata_exact=all(ca[k]==cb[k] for k in ('noise_seed','reward','success','terminated','truncated','sampling','native_run_steps','native_reward_step','public_state_after','exposed_state_after'))))
        fast_full.append(dict(case_index=ci,checks=checks,chunks=chunks,
            all_recorded_comparisons_exact=all(checks.values()) and all(all(v for k,v in c.items() if k!='chunk') for c in chunks)))
    paired = {}
    for left,right in [('native10','port10'),('port10','port5'),('port5','sde5_full'),('sde5_full','sde5_fast')]:
        counts = Counter(f"{int(episodes[i,left]['success'])},{int(episodes[i,right]['success'])}" for i in range(16))
        paired[left+'__'+right] = dict(contingency=dict(counts),
            left_success_cases=[i for i in range(16) if episodes[i,left]['success']],
            right_success_cases=[i for i in range(16) if episodes[i,right]['success']])
    files = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix not in ('.json','.npy','.npz','.jpg','.png'): continue
        if p.name in ('READOUT.json','INITIAL-INTEGRITY.json'): continue
        files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    assert result_sha == sha(root/'RESULT.json') and exit_sha == sha(root/'EXIT.json')
    integrity = dict(status='passed',result_sha256=result_sha,exit_sha256=exit_sha,
        episode_files_independently_reopened=80,initial_identities_paired_cases=16,
        native_frozen_query_receipts=native_receipts,optimizer_updates=0,
        files=files,total_files=len(files),total_bytes=sum(f['bytes'] for f in files),checked_unix=time.time())
    readout = dict(status='complete',summaries=summaries,paired_success=paired,
        requested_controls=requested,full_vs_fast=fast_full,
        all_full_vs_fast_recorded_comparisons_exact=all(c['all_recorded_comparisons_exact'] for c in fast_full),
        batch_size=1,horizon=400,cases=report['cases'],worker_gpu_seconds=exit_record['worker_gpu_seconds'],
        physical_duration_measured=False,fast_executed_controls_measured=False,
        interpretation='Fixed16-case inference diagnostic through RLinf controller/generated prompt; not an official Sidney benchmark, batch64 collector reproduction, or final trained-policy comparison')
    save(root/'INITIAL-INTEGRITY.json',integrity); save(root/'READOUT.json',readout)
    print(json.dumps(dict(status='passed',summaries=summaries,
        all_full_vs_fast_recorded_comparisons_exact=readout['all_full_vs_fast_recorded_comparisons_exact'],
        paired_success=paired,total_files=len(files),total_bytes=integrity['total_bytes'],
        worker_gpu_seconds=exit_record['worker_gpu_seconds'])))


if __name__ == '__main__': main()
