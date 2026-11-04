"""Replay every frozen diagnostic query on CPUs and inspect actual reset inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time


def main():
    assert sys.platform=='linux'
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.vla_history import FrozenHistoryController,load_policy
    torch.set_num_threads(1)
    run=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
    group=run/'gpu/direct-snapshot-ID-qualification-01'
    top=json.loads((group/'RESULT.json').read_text())
    assert top['completed'] and top['qualification_only']
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    began=time.time();methods={}
    for method,reference in top['selected_candidates'].items():
        policy,_=load_policy(reference['path'],expected_sha256=reference['sha256'],device='cpu')
        original_step=policy.step;seen=[]
        def checked_step(token,state=None,**kwargs):
            d=policy.config.observation_dim
            if token[0,-1].item()==1 and mode=='episode':
                assert state is None
                assert torch.count_nonzero(token[:,d:-1]).item()==0
                seen.append('cleared')
            return original_step(token,state,**kwargs)
        policy.step=checked_step
        maximum=0.;queries=0;episodes=0
        banks=[Path(row['path']) for row in top['source_banks']
            if json.loads(Path(row['path']).read_text())['method']==method]
        for bank_path in banks:
            bank=json.loads(bank_path.read_text());mode=bank['mode'];by_case={}
            for ref in bank['references']:
                assert sha(ref['path'])==ref['sha256']
                saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
                by_case.setdefault(json.dumps(saved['case'],sort_keys=True),[]).append(saved)
            for attempts in by_case.values():
                controller=FrozenHistoryController(policy,mode=mode)
                assert sorted(row['attempt'] for row in attempts)==list(range(1,7))
                for saved in sorted(attempts,key=lambda row:row['attempt']):
                    for chunk,ref in enumerate(saved['chunks']):
                        assert sha(ref['path'])==ref['sha256']
                        with gzip.open(ref['path'],'rb') as stream:row=pickle.load(stream)
                        control=controller.act(row['features'],deterministic=True)
                        error=float(np.max(np.abs(control-row['control'])))
                        maximum=max(maximum,error)
                        # CPU and CUDA FP32 attention/GEMM reductions differ.
                        # This numerical check is distinct from the exact
                        # cleared-input/cache assertions and raw command audit.
                        np.testing.assert_allclose(control,row['control'],rtol=1e-4,atol=1e-4)
                        expected=chunk+1 if mode=='episode' else (saved['attempt']-1)*12+chunk+1
                        assert int(controller.state['length'][0])==expected
                        controller.observe(row['executed_control'],row['reward'],row['done'])
                        queries+=1
                    episodes+=1
        assert len(seen)==36 and queries==864 and episodes==72
        methods[method]=dict(passed=True,queries=queries,episodes=episodes,
            inspected_episode_reset_tokens=len(seen),maximum_CPU_vs_saved_GPU_control_error=maximum,
            relative_tolerance=1e-4,absolute_tolerance=1e-4,
            prior_action_reward_done_zeroed_and_cache_None_on_reset=True,
            post_query_cache_lengths_match_declared_mode=True)
        print(json.dumps(dict(method=method,**methods[method])),flush=True)
    result=dict(passed=True,methods=methods,episodes=288,queries=3456,
        GPU_hours=0,CPU_only=True,SGD=0,elapsed_seconds=time.time()-began,
        group_result_sha256=sha(group/'RESULT.json'),
        initial_precision_check='1e-5 CPU/CUDA output tolerance failed: seven of600 coordinates on one query differed by up to1.6868114e-5; retain this as a numerical precision observation, not a rollout failure.',
        diagnosis='Saved memory_entry is a pre-act boundary. Episode reset happens inside act; all actual saved GPU outputs replay within1e-4 CPU tolerances with exact cleared causal inputs. This is numerical agreement, not byte-exact cross-device replay. No rollout or policy change.')
    with (run/'SNAPSHOT-RESET-REPLAY.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
