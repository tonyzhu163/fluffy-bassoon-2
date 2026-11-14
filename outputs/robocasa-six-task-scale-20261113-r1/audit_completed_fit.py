"""Audit a real fixed-final fit before binding it into the matched readout.

Checks protect the scientific endpoint (10000 updates), exact exported weights,
and intact finite recovery state. No simulator episode or GPU forward is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', type=Path, required=True)
    parser.add_argument('--method', choices=['ad', 'dpt', 'dit', 'ic_iql'], required=True)
    parser.add_argument('--corpus-size', type=int, choices=[3, 6], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    import torch
    torch.set_num_threads(4)
    assert not args.output.exists()
    complete = json.loads(args.complete.read_text())
    terminal_path = args.complete.parent.parent / (args.method + '-EXIT.json')
    terminal = json.loads(terminal_path.read_text())
    assert terminal['method'] == args.method and terminal['exit_code'] == 0 and terminal['complete']
    progress_path = args.complete.with_name('PROGRESS.json')
    progress = json.loads(progress_path.read_text())
    contract = json.loads(args.complete.with_name('CONTRACT.json').read_text())
    assert complete['completed'] and complete['native_parameters_unchanged']
    assert progress['complete'] and progress['phase'] == 'actor'
    assert complete['progress']['actor_update'] == progress['actor_update'] == 10000
    assert complete['contract'] == contract
    assert contract['source_commit'] == '4987585e7427061de30771ee207f9b2b79eedc6a'
    assert contract['objective'] == args.method and contract['corpus_size'] == args.corpus_size
    assert contract['corpus_spec_sha256'] == '85394094fcc159984ffbaf249c93cbced236b723ce4fd6d39fa3e54b5dce4e2e'
    assert contract['seed'] == 0 and contract['logical_batch'] == 8
    assert contract['checkpoint_selection'] == 'fixed final' and contract['actor_updates'] == 10000
    expected_values = 1000 if args.method in ('dit', 'ic_iql') else 0
    assert contract['value_updates'] == progress['value_update'] == expected_values
    for ref in [progress['checkpoint'], complete['candidate']]:
        assert digest(ref['path']) == ref['sha256']
    checkpoint = torch.load(progress['checkpoint']['path'], map_location='cpu', weights_only=False)
    candidate = torch.load(complete['candidate']['path'], map_location='cpu', weights_only=False)
    assert candidate['format'] == 'flow_steering_bridge_policy_v1'
    assert checkpoint['format'] == 'matched_kv_fit_v1' and checkpoint['contract'] == contract
    assert checkpoint['progress']['actor_update'] == 10000
    assert checkpoint['progress']['value_update'] == expected_values
    assert candidate['manifest']['update'] == 10000
    assert candidate['manifest']['contract'] == dict(contract, config=complete['config'])
    assert candidate['config'] == complete['config']
    assert candidate['policy'].keys() == checkpoint['policy'].keys()
    for name, value in candidate['policy'].items():
        assert torch.isfinite(value).all() and torch.equal(value, checkpoint['policy'][name]), name

    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
        elif isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite_tree(item)

    optimizer_steps = {}
    for key, expected in [('optimizer', 10000), ('value_optimizer', expected_values)]:
        value = checkpoint[key]
        if expected == 0:
            assert value is None and checkpoint['values'] is None
            continue
        assert value and value['state']
        finite_tree(value)
        steps = {int(state['step'].item()) for state in value['state'].values()}
        assert steps == {expected}, (key, steps)
        assert all(group['lr'] == contract['learning_rate'] and group['weight_decay'] == 0 for group in value['param_groups'])
        optimizer_steps[key] = dict(steps=sorted(steps), state_entries=len(value['state']))
    finite_tree(checkpoint['values'])
    rng = checkpoint['rng']
    assert all(key in rng for key in ['sampler', 'python', 'numpy', 'torch', 'cuda'])
    assert len(rng['cuda']) == 1 and checkpoint['flow_generator'].numel() > 0
    result = dict(passed=True, method=args.method, corpus_size=args.corpus_size,
        complete=dict(path=str(args.complete), sha256=digest(args.complete)),
        terminal=dict(path=str(terminal_path), sha256=digest(terminal_path)),
        progress=dict(path=str(progress_path), sha256=digest(progress_path)),
        candidate=complete['candidate'], checkpoint=progress['checkpoint'],
        actor_updates=10000, value_updates=expected_values, policy_tensors=len(candidate['policy']),
        export_exactly_matches_final_checkpoint=True, optimizer_steps=optimizer_steps,
        policy_values_and_optimizers_finite=True, recovery_RNG_fields_present=True,
        native_parameters_unchanged=complete['native_parameters_unchanged'],
        execution='Baige development Linux CPU; no model forward or GPU computation',
        hostname=socket.gethostname(), checked_unix=time.time(), GPU_hours=0,
        audit_script_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
