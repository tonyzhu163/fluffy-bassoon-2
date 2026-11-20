"""Summarize unique native demonstrations and endpoint outcomes from the shared run directory."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    sig=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            sig.update(block)
    return sig.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    counts=defaultdict(list)
    result={"jobs":{},"source":{},"fits":{}}
    for job in sorted(args.run.iterdir()):
        if not job.is_dir() or not (job/"launch.txt").exists():
            continue
        pin,host,pid,start=(job/"launch.txt").read_text().split()
        receipt=dict(pin=pin,host=host,start_unix=int(start))
        if (job/"exit.txt").exists():
            rc,stop,begin=map(int,(job/"exit.txt").read_text().split())
            receipt.update(exit_code=rc,end_unix=stop,worker_seconds=stop-begin)
        result["jobs"][job.name]=receipt
        index=job/"bank/index.json"
        if index.exists():
            raw=json.loads(index.read_text())
            accepted=[r for r in raw["attempts"] if r.get("accepted")]
            receipt.update(accepted=len(accepted), searchable_attempts=len(raw["attempts"]),index_sha256=sha(index))
            for r in accepted:
                assert r['native_success'] and r['terminal_reward']>0 and r['done']
                path=job/'bank'/r['file']
                if not path.is_file():
                    raise FileNotFoundError(path)
                counts[r['body']].append(dict(job=job.name,pose=r['pose'],seed=r['seed'],steps=r['steps'],
                           file=r['file'],reward=r['terminal_reward'],
                           released_row=r['reset']['environment_reproducibility']['released_row_fingerprint'],
                           initial_state=r['reset']['environment_reproducibility']['initial_state_fingerprint']))
        if (job/'fit/manifest.json').exists():
            fit={'manifest_sha256':sha(job/'fit/manifest.json'),
                 'training_coverage':json.loads((job/'fit/manifest.json').read_text())['contract']['coverage']}
            for stage in ['dev','confirm']:
                r=job/'fit'/stage/'report.json'
                if r.exists():
                    d=json.loads(r.read_text());fit[stage]={k:d.get(k) for k in ['status','source_gate','selected_step','motor_checkpoint_sha256','scores','native_successes']}
            r=job/'ur5e/report.json'
            if r.exists():
                d=json.loads(r.read_text());fit['ur5e']={k:d.get(k) for k in ['status','native_successes','terminal_rewards','versus_raw','correct_versus']}
            result['fits'][job.name]=fit
    for body,rows in counts.items():
        result['source'][body]={}
        for split,subset in [('training',[r for r in rows if r['pose']<28]),('offline_validation',[r for r in rows if r['pose']>=28])]:
            result['source'][body][split]=dict(accepted_tapes=len(subset),layout_ids=sorted({r['pose'] for r in subset}),
                         unique_native_initial_states=len({r['initial_state'] for r in subset}),controls=sum(r['steps'] for r in subset))
    result['total_completed_worker_gpu_hours']=sum(j.get('worker_seconds',0) for j in result['jobs'].values())/3600
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=="__main__":
    main()
