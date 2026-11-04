"""Audit independent completed native banks concurrently on allocated CPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    assert sys.platform == 'linux'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8 and not args.output.exists()
    top = json.loads((args.group/'RESULT.json').read_text())
    assert top['completed'] and top['deployment_optimizer_updates'] == 0
    banks = sorted(args.group.glob('node-*/*/*/bank-*/RESULT.json'))
    if not banks:
        banks = sorted(args.group.glob('*/*/bank-*/RESULT.json'))
    assert banks and len(set(banks)) == len(banks)
    # Import CPU libraries before starting threads; no model or CUDA context.
    import cv2
    import numpy
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    script = str(Path(__file__).with_name('AUDIT-ROLLOUT.py'))
    spec = importlib.util.spec_from_file_location('native_bank_audit',script)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args.output.mkdir(parents=True)
    began=time.time()
    def one(bank):
        target=args.output/bank.relative_to(args.group).with_name('AUDIT.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        audit=module.audit_result(bank,target)
        return dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            **{key:audit[key] for key in ('episodes','physical_controls','complete_chunks','decoded_videos','video_frames')})
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,bank) for bank in banks]
        for future in as_completed(futures):
            rows.append(future.result())
            payload=dict(completed_banks=len(rows),total_banks=len(banks),elapsed_seconds=time.time()-began)
            temp=args.output/'PROGRESS.tmp';temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(args.output/'PROGRESS.json')
    summary=dict(passed=True,group=str(args.group),group_result_sha256=hashlib.sha256((args.group/'RESULT.json').read_bytes()).hexdigest(),
        banks=len(rows),bank_audits=sorted(rows,key=lambda row:row['path']),CPU_only=True,GPU_hours=0,
        elapsed_seconds=time.time()-began,workers=args.workers,
        **{key:sum(row[key] for row in rows) for key in ('episodes','physical_controls','complete_chunks','decoded_videos','video_frames')})
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='bank_audits'}))


if __name__ == '__main__':
    main()
