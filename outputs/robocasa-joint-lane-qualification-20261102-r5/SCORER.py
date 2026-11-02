"""One persistent official engine beside the native lane worker; no SGD."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import write_new, flora_jobs, sha
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol
from fluffy_bassoon.baige_gpu import verify_selected_cuda
import torch

def main():
    out = Path(os.environ['TASK_OUT'])
    code = Path(os.environ['TASK_META'])
    uuid = verify_selected_cuda(torch)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):
        os.environ.pop(key, None)
    spec = importlib.util.spec_from_file_location('scoring_service', code/'scripts/diagnostics/robocasa_residual_score.py')
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    started = time.monotonic()
    worker = TerminalWorker(os.environ['TASK_FLORA_MODEL'], batch_size=8, gpu_memory_utilization=.75)
    gate=json.loads((Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-five-task-fifty-score-20261102-r1/exchange/cohorts/CoffeeSetupMug/baseline-validation.json')).read_text())
    ref=gate['native_episodes']; assert sha(ref['path'])==ref['sha256']
    episodes=json.loads(Path(ref['path']).read_text())[:8]
    tick=time.monotonic()
    worker.score_jobs(flora_jobs(episodes,protocol('CoffeeSetupMug',baseline_count=50)),out/'warmup-scores.jsonl')
    warm_seconds=time.monotonic()-tick
    raw = subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    write_new(out/'engine-ready.json', dict(cold_seconds=time.monotonic()-started, uuid=uuid,
        warmup_score_seconds=warm_seconds, warmup_videos=8, memory_mib=[float(x) for x in raw.strip().split(',')], batch_size=8, gpu_memory_utilization=.75))
    progress = SimpleNamespace(emit=lambda **kw: print(json.dumps(kw), flush=True))
    progress = SimpleNamespace(emit=lambda **kw: print(json.dumps(kw), flush=True))
    while not (out/'shutdown.json').exists():
        for choice in sorted(out.glob('selection-*.json')):
            selected=json.loads(choice.read_text());lanes=selected['lanes']
            profile=dict(task='joint_five',source_commit=os.environ['TASK_CODE_PIN'],annotation_rank=0,
                F_output=str(out/f'lanes-{lanes}'),stream_scoring=True)
            scorer_out=out/f'scorer-{lanes}';scorer_out.mkdir(exist_ok=True)
            result=out/f'scoring-result-{lanes}.json'
            if result.exists():continue
            service.prescore(profile,worker,scorer_out,progress)
            request_path=out/f'completed-bank-{lanes}.json'
            if request_path.exists():
                request=json.loads(request_path.read_text());episodes=request['episodes']
                mapping={e['id']:protocol(case['task'],baseline_count=50) for e,case in zip(episodes,request['cases'],strict=True)}
                tick=time.monotonic()
                rows,labels,receipt=service.score(worker,episodes,scorer_out/'final.jsonl',
                    selected['model_receipt'],episode_protocols=mapping,cache=scorer_out/'prescore')
                write_new(result,dict(completed=True,labels=len(labels),final_label_wait_seconds=time.monotonic()-tick,
                    receipt=receipt,model_load_count=worker.model_load_count,completed_batches=worker.completed_batches))
        time.sleep(.5)

if __name__ == '__main__':
    main()
