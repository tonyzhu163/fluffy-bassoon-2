"""Small frozen native replay; no simulator evaluation panel or SGD."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

MODEL=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-qualification-20261105-r1')
PYTHON='/mowenjie/fluffy-bassoon-2/envs/lerobot-sidney-92f96f33-20261105-r2/bin/python'
LEROBOT='/mowenjie/fluffy-bassoon-2/sources/lerobot-sidney-92f96f33-20261105-r1/src'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vectors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',choices=['bfloat16','float32'],default='bfloat16')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    worker_code=Path(__file__).resolve().parents[1]/'robotwin-sidney-qualification-20261105-r1/model_worker.py'
    env=dict(os.environ,PYTHONPATH=LEROBOT,TOKENIZERS_PARALLELISM='false')
    with (args.output/'worker.log').open('w') as log:
        worker=subprocess.Popen([PYTHON,str(worker_code),'--model',str(MODEL),
            '--output',str(args.output),'--precision',args.precision],stdin=subprocess.PIPE,
            stdout=log,stderr=subprocess.STDOUT,text=True,env=env)
        def wait(path):
            deadline=time.monotonic()+420
            while not path.exists():
                if worker.poll() is not None:raise RuntimeError(f'Native worker exited{worker.returncode}; {args.output}/worker.log')
                if time.monotonic()>deadline:raise TimeoutError(str(path))
                time.sleep(.25)
        try:
            wait(args.output/'READY.json')
            for i,source in enumerate(sorted(args.vectors.glob('*.npz'))):
                target=args.output/(source.stem+'-actions.npz')
                worker.stdin.write(json.dumps(dict(observation=str(source),actions=str(target),
                    noise_seed=1000+i,check_queue_parity=False))+'\n');worker.stdin.flush()
                wait(target.with_suffix('.json'))
                print(json.dumps(dict(vector=source.name,precision=args.precision,status='complete')),flush=True)
        finally:
            worker.stdin.close()
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=15)
        assert worker.returncode==0


if __name__=='__main__':main()
