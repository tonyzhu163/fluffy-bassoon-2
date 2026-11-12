"""Complete missing branches against immutable saved baseline/candidate records."""
import argparse
import gc
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np

from collect import (BUDGET, QUERIES, SEEDS, Motor, identity, picture, propose, reset,
                     stable_counter_regions, state, write)
from render_gc import bind_context_cleanup


def main():
    ap = argparse.ArgumentParser()
    for name in ('source','output','checkpoint'):
        ap.add_argument('--'+name,type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.time()
    def event(phase,**fields):
        row = dict(phase=phase,elapsed_seconds=time.time()-started,**fields)
        with (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    from robocasa.utils.env_utils import convert_action
    bind_context_cleanup()
    gc.disable()
    stable_counter_regions()
    completed = json.loads((args.source/'branches.json').read_text())
    completed_keys = {(r['seed'],r['query'],r['candidate']) for r in completed}
    pending = [(s,q,c) for s in SEEDS for q in QUERIES for c in (0,1) if (s,q,c) not in completed_keys]
    write(args.output/'source.json',dict(source=str(args.source),completed=len(completed),pending=pending))
    baselines = {}
    for seed in sorted({r[0] for r in pending}):
        source = args.source/str(seed)
        prior = json.loads((source/'baseline.json').read_text())
        with np.load(source/'baseline.npz',allow_pickle=False) as arrays:
            values = {k:arrays[k].copy() for k in arrays.files}
        with (source/'global-rng.pkl').open('rb') as stream:
            rngs = pickle.load(stream)
        baselines[seed] = prior,values,rngs
        dest = args.output/str(seed)
        dest.mkdir()
        # Diagnose the previous initial-state rejection before loading a motor
        # or executing any candidate. Never weaken the component comparisons.
        wrapped,obs = reset(seed)
        try:
            current = identity(wrapped,obs)
            check = dict(expected=prior['initial'],actual=current,
                differing=[k for k in current if current[k] != prior['initial'][k]])
            write(dest/'initial-check.json',check)
            picture(dest/'initial-check.png',obs,'Fresh initial replay check')
            np.testing.assert_array_equal(state(obs),values['states'][0])
            assert not check['differing'], check
            event('initial_verified',seed=seed)
        finally:
            wrapped.close()
            del wrapped
            gc.collect()
    policy = Motor(args.checkpoint)
    event('loaded',policy_pid=policy.process.pid)
    results = []
    try:
        for seed,q,candidate in pending:
            prior,values,rngs = baselines[seed]
            anchors = {int(k):v for k,v in prior['anchors'].items()}
            commands,states = values['commands'],values['states']
            rewards,successes = values['rewards'].tolist(),values['successes'].tolist()
            dest = args.output/str(seed)
            wrapped,obs = reset(seed)
            delivered,earned,success = [],[],[]
            try:
                actual = identity(wrapped,obs)
                if actual != prior['initial']:
                    write(dest/f'initial-mismatch-{q}-{candidate}.json',dict(expected=prior['initial'],actual=actual,
                        differing=[k for k in actual if actual[k] != prior['initial'][k]]))
                    raise AssertionError('Initial state differs from immutable baseline')
                for step,action in enumerate(commands[:q]):
                    obs,_,_,_,_ = wrapped.step(convert_action(action))
                    np.testing.assert_array_equal(state(obs),states[step+1])
                random.setstate(rngs[q][0]); np.random.set_state(rngs[q][1])
                assert identity(wrapped,obs) == anchors[q], ('fork',seed,q,candidate)
                for step in range(q,BUDGET,5):
                    if candidate == 0:
                        assert identity(wrapped,obs) == anchors[step], ('continuation',seed,q,step)
                    action_chunk = (values['candidates'][QUERIES.index(q),candidate] if step == q
                                    else propose(policy,obs,seed,step))
                    if candidate == 0:
                        np.testing.assert_array_equal(action_chunk[:5],commands[step:step+5])
                    for action in action_chunk[:5]:
                        obs,reward,done,truncated,info = wrapped.step(convert_action(action))
                        assert not done and not truncated
                        delivered.append(action.copy()); earned.append(float(reward)); success.append(bool(info['success']))
                if candidate == 0:
                    assert identity(wrapped,obs) == anchors[BUDGET]
                    assert earned == rewards[q:] and success == successes[q:]
                np.savez_compressed(dest/f'branch-{q}-{candidate}.npz',commands=delivered,rewards=earned,successes=success)
                picture(dest/f'branch-{q}-{candidate}-final.png',obs,f'Final {seed} {q} {candidate}')
                result = dict(seed=seed,query=q,candidate=candidate,fork_exact=True,global_rng_restored=True,
                    prefix_state_checks=q,baseline_continuation_exact=candidate==0,
                    success_before_query=any(successes[:q]),success_once=any(success),final_success=success[-1],
                    native_return=sum(earned),controls=BUDGET-q,baseline_source=str(args.source))
                write(dest/f'branch-{q}-{candidate}.json',result)
                results.append(result)
                write(args.output/'branches.json',results)
                event('branch_complete',**result)
            finally:
                wrapped.close()
                del wrapped
                gc.collect()
    finally:
        policy.close()
    write(args.output/'complete.json',dict(source=str(args.source),branches=results,wall_seconds=time.time()-started))
    event('complete')


if __name__ == '__main__':
    main()
