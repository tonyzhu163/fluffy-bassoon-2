"""Measure the eight immutable selected steering candidates without robot actions."""
import argparse
import gc
import json
from pathlib import Path
import time

from RUNTIME import Runtime, PARENT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('code', 'probe', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    contract_path = Path(__file__).with_name('CONTRACT.json')
    contract = json.loads(contract_path.read_text())
    runtime = Runtime(args.code, args.commit, args.probe, contract)
    import torch
    from fluffy_bassoon.meta_rl.flow_history import load_bridge_policy
    from fluffy_bassoon.meta_rl.flow_bridge_train import steering_effect, steering_gate
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _json
    index_path = PARENT / 'finalized-fits-20261106-r1/INDEX.json'
    index = json.loads(index_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for objective in contract['diagnostic']['objectives']:
        data, manifest = runtime.data(objective, 'validation')
        prepare = runtime.prepare(data, manifest)
        for interface in contract['diagnostic']['interfaces']:
            key = interface + '/' + objective
            reference = index['models'][key]['artifacts']['candidate.pt']
            candidate = Path(reference['path'])
            policy, exported = load_bridge_policy(candidate, expected_sha256=reference['sha256'], device='cuda:0')
            binding = exported['contract']
            if (binding['interface'] != interface or binding['objective'] != objective
                    or binding['assembly_sha256'] != contract['assembly_sha256']
                    or binding['native_model_sha256'] != contract['native_model_sha256']
                    or binding['native_norm_sha256'] != contract['native_norm_sha256']
                    or binding['validation_sha256'] != runtime.assembly['datasets'][objective + '_validation']['sha256']):
                raise ValueError('Candidate differs from original frozen-expert/data identities')
            before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            batches = runtime.validation_plan(policy, data, objective)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); began = time.monotonic()
            effect = steering_effect(policy, runtime.native.model, batches, prepare)
            torch.cuda.synchronize()
            runtime.check_frozen()
            if any(not torch.equal(v, policy.state_dict()[k]) for k, v in before.items()):
                raise ValueError('Candidate changed during measurement')
            row = dict(model=key, source_commit=args.commit, candidate_sha256=reference['sha256'],
                candidate_update=exported['update'], diagnostic_contract_sha256=runtime.sha(contract_path),
                validation_sha256=binding['validation_sha256'], native_model_sha256=binding['native_model_sha256'],
                effect=effect, gate=steering_gate(effect, **contract['steering_gate']),
                elapsed_seconds=time.monotonic() - began, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
                unchanged_candidate=True, unchanged_native=True, simulator_controls=0, optimizer_updates=0,
                existing_readout_admission=False)
            _json(args.output / interface / (objective + '.json'), row)
            results[key] = row
            _json(args.output / 'RESULT.json', dict(completed=len(results) == 8, models=results,
                contract_sha256=runtime.sha(contract_path), source_commit=args.commit,
                finalized_index_sha256=runtime.sha(index_path)))
            print(json.dumps(row, allow_nan=False), flush=True)
            del policy, before, batches
            gc.collect(); torch.cuda.empty_cache()
        del data, manifest
        gc.collect()
    runtime.reader.close()


if __name__ == '__main__':
    main()
