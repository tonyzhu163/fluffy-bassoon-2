"""Allocated-Linux synthetic end-to-end fits, exports and operational resume."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('Qualification computation runs only on allocated Linux')
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.frozen_steering_history import save_corpus, split_streams
    from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_policy, FrozenHistoryController
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'tests'))
    from test_frozen_steering_history import source, all_episodes, teacher_file
    args.output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True).strip():
        raise ValueError('Require clean pinned source')
    results = []
    started = time.monotonic()
    for arm in ('velocity', 'kv_all_layers'):
        target = args.output/arm
        target.mkdir()
        spec = source(arm)
        streams = split_streams(all_episodes(target/'episodes', spec), spec,
            validation_slots=[2], expected_updates=(0,1))
        manifests = save_corpus(target/'corpus', streams, spec,
            teacher_reference=teacher_file(target,spec), provenance=dict(synthetic_qualification=True))
        for method in ('ad','dpt','dit','ic_iql'):
            module = 'history_value_train' if method in ('dit','ic_iql') else 'vla_history_train'
            common = [sys.executable,'-B','-m','fluffy_bassoon.meta_rl.'+module]
            for split in ('train','validation'):
                common += ['--'+split+'-data',str(target/'corpus'/f'{method}-{split}.npz'),
                    '--'+split+'-sha256',manifests[method+'_'+split]['sha256']]
            common += ['--width','8','--layers','1','--heads','2','--context-length','16',
                '--max-episodes-per-window','2','--batch-size','2','--eval-every','1',
                '--validation-batches','1','--threads','1','--seed','31','--device',args.device,
                '--source-commit',commit,'--expected-hostname',socket.gethostname(),
                '--until-converged','--minimum-updates','1','--validation-patience','1',
                '--validation-relative-delta','1000']
            if method in ('dit','ic_iql'):
                common += ['--objective',method]
            output = target/method
            def fit(where, seconds, resume=False):
                cmd = common+['--output-dir',str(where),'--max-elapsed-seconds',str(seconds)]
                if resume: cmd += ['--resume',str(where/'latest.pt')]
                log = target/(where.name+('-resume' if resume else '')+'.log')
                with log.open('w') as stream:
                    subprocess.run(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
            fit(output, 120)
            status = json.loads((output/'training-status.json').read_text())
            assert status['converged'],status
            candidate = output/'candidate.pt'
            policy, manifest = load_policy(candidate,expected_sha256=file_sha256(candidate))
            assert policy.config.noise_dim == spec.steering_dim
            controller = FrozenHistoryController(policy)
            proposal = controller.act(np.ones(spec.feature_dim,np.float32),deterministic=True)
            assert proposal.shape==(spec.steering_dim,) and np.isfinite(proposal).all()
            controller.observe(proposal,1.,True)
            row = dict(arm=arm,method=method,converged=True,candidate_sha256=file_sha256(candidate),
                action_dimension=spec.steering_dim,actual_simulator_episodes=0)
            if method in ('dit','ic_iql'):
                recovery = target/(method+'-resumed')
                fit(recovery, 1e-9)
                partial = json.loads((recovery/'training-status.json').read_text())
                assert not partial['converged'] and partial['updates']['value']==1
                fit(recovery,120,resume=True)
                resumed = torch.load(recovery/'latest.pt',map_location='cpu',weights_only=False)
                original = torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
                assert resumed['converged'] and resumed['updates']==original['updates']
                for key,value in original['learner'].items():
                    torch.testing.assert_close(value,resumed['learner'][key],rtol=0,atol=0)
                row['interrupted_value_stage_resume_all_weights_exact']=True
                row['updates']=original['updates']
            results.append(row)
    report=dict(passed=True,source_commit=commit,hostname=socket.gethostname(),device=args.device,
        elapsed_seconds=time.monotonic()-started,actual_simulator_episodes=0,results=results,
        interpretation='Synthetic fit/serialization/resume qualification only; not robot performance or converged scientific students')
    (args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
