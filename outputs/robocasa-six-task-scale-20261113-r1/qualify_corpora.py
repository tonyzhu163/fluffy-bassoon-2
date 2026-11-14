"""Check actual corpus/source-query alignment and sampler resumption on Linux CPU."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

from corpus import balanced_batch, load_corpus
from labels import sha, write

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--corpora', type=Path, required=True)
    parser.add_argument('--corpora-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and not args.output.exists()
    sys.path[:0] = [str(CODE / 'src'), os.environ['RLINF_PATH']]
    import numpy as np
    import torch
    from fluffy_bassoon.rlinf.native_ppo import checked_checkout
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowBridgeConfig, HistoryFlowSteeringPolicy, sample_bridge_query_batch
    from fluffy_bassoon.meta_rl.flow_source_queries import FlowSourceQueries
    checked_checkout(CODE, args.pin)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    # Only the public sampler's dimensions/device are used; no policy forward,
    # native VLA, optimization, or accelerator qualification occurs here.
    policy = HistoryFlowSteeringPolicy(FlowBridgeConfig(width=256, layers=6, heads=8, context_length=192))
    norm = Path('/mowenjie/fluffy-bassoon-2/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json')
    stats = json.loads(norm.read_text())
    stats = stats.get('norm_stats', stats)['actions']
    checked = []
    for size in (3, 6):
        for objective in ('ad', 'dpt', 'dit', 'ic_iql'):
            corpus = load_corpus(args.corpora, args.corpora_sha256, size, objective)
            assert sha(norm) == corpus['identity']['native_norm_sha256']
            codec = DirectActionCodec(tuple(corpus['identity']['codec']['mean']), tuple(corpus['identity']['codec']['std']))
            reader = FlowSourceQueries(corpus['sources'], codec, cached_members=1)
            try:
                for split, streams in [('train', 112), ('validation', 16)]:
                    rng = np.random.default_rng(734001)
                    state = copy.deepcopy(rng.bit_generator.state)
                    kwargs = dict(task_count=size, streams_per_task=streams, batch_size=8,
                        max_episodes=12, discount=.99, action_codec=codec,
                        native_action_stats=stats, objective=objective)
                    trials, ids = corpus['data'][split], corpus['ids'][split]
                    batch = balanced_batch(sample_bridge_query_batch, trials, policy, rng, **kwargs)
                    after = copy.deepcopy(rng.bit_generator.state)
                    rng.bit_generator.state = state
                    restored = balanced_batch(sample_bridge_query_batch, trials, policy, rng, **kwargs)
                    assert rng.bit_generator.state == after and batch.keys() == restored.keys()
                    for field, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            assert torch.equal(value, restored[field])
                        else:
                            assert value == restored[field]
                    observations = reader.observations(batch['query_indices'], trials, ids)
                    expected = np.stack([trials[i]['observations'][t, :25] for i, t in batch['query_indices']])
                    assert np.array_equal(np.asarray(observations['states'], np.float32), expected)
                    assert batch['target_actions'].shape == (8, 50, 32)
                    checked.append(dict(corpus_size=size, objective=objective, split=split,
                        trials=len(trials), sampled_queries=[dict(stream=ids[i], decision=t) for i, t in batch['query_indices']],
                        raw_states_behavior_versions_match=True, sampler_resume_exact=True))
            finally:
                reader.close()
            del corpus, trials, reader, batch, restored
            gc.collect()
    write(args.output, dict(passed=True, CPU_only=True, source_commit=args.pin,
        corpora=dict(path=str(args.corpora), sha256=args.corpora_sha256),
        sampler_sha256=sha(HERE / 'corpus.py'), checked_cells=checked, sampled_queries=128,
        hostname=socket.gethostname(), checked_unix=time.time(),
        scope='All eight objective/corpus loaders and disjoint splits;128 actual RGB queries checked against states, behavior actions and producer versions; sampled windows resume exactly from sampler RNG. No native GPU model or gradient qualification claimed.'))
    print(json.dumps(dict(passed=True, checked_cells=len(checked), sampled_queries=128)), flush=True)


if __name__ == '__main__':
    main()
