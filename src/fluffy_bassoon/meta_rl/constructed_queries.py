"""Measured DPT query labels and an actuator-independent categorical student.

`python -m fluffy_bassoon.meta_rl.constructed_queries PUBLIC.npz BRANCHES.json OUT.npz`
exports one matched trial. Candidate order is fixed for an actuator. Training uses
one classification target at the fresh query, after reading the support episode.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .causal_decoder import CausalDecoderCore
from .vla_history import trial_tokens


def public_tokens(support_obs, delivered_actions, native_rewards, query_obs):
    """Make one complete support episode followed by its fresh query observation."""
    obs, action, reward, query = map(np.asarray, (support_obs, delivered_actions, native_rewards, query_obs))
    if (obs.ndim != 2 or min(obs.shape) == 0 or action.ndim != 2 or action.shape[0] != len(obs)
            or not action.shape[1] or reward.shape != (len(obs),) or query.shape != obs.shape[1:]
            or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (obs, action, reward, query))):
        raise ValueError('Expected aligned finite float32 support transitions and one query feature vector')
    starts = np.zeros(len(obs)+1, bool); starts[[0, -1]] = True
    dones = np.zeros_like(starts); dones[-2] = True
    return trial_tokens(dict(observations=np.vstack((obs, query)),
                            executed_noise=np.vstack((action, np.zeros_like(action[:1]))),
                            rewards=np.append(reward, np.float32(0)), dones=dones, episode_starts=starts))


def measured_label(candidates, branches, *, trial_id, decision_sha256):
    """Native success first, smallest absolute miss among successes second; abstain on ties."""
    candidates = tuple(candidates)
    if (len(candidates) < 2 or any(type(c) is not str or not c for c in candidates)
            or len(set(candidates)) != len(candidates) or len(branches) != len(candidates)):
        raise ValueError('Require one measurement per distinct ordered candidate')
    if not trial_id or len(decision_sha256) != 64 or any(c not in '0123456789abcdef' for c in decision_sha256):
        raise ValueError('Require trial identity and the decision observation hash')
    outcomes = {}
    for row in branches:
        if (row['trial_id'] != trial_id or row['decision_sha256'] != decision_sha256
                or row['restore_verified'] is not True):
            raise ValueError('Every measured branch must match the verified query and trial')
        candidate, success, miss = row['candidate'], row['native_success'], row['signed_miss_m']
        if (candidate not in candidates or candidate in outcomes or type(success) is not bool
                or type(miss) not in (int, float) or not np.isfinite(miss)):
            raise ValueError('Invalid or duplicate branch measurement')
        outcomes[candidate] = (success, abs(miss))
    wins = [i for i, c in enumerate(candidates) if outcomes[c][0]]
    if not wins:
        return None
    distance = min(outcomes[candidates[i]][1] for i in wins)
    best = [i for i in wins if outcomes[candidates[i]][1] == distance]
    return best[0] if len(best) == 1 else None


def export_query(public, evidence, output):
    with np.load(public, allow_pickle=False) as src:
        trial, decision = str(src['trial_id'].item()), str(src['decision_sha256'].item())
        candidates = src['candidate_ids'].tolist()
        tokens = public_tokens(src['support_obs'], src['delivered_actions'], src['native_rewards'], src['query_obs'])
    rows = json.loads(Path(evidence).read_text())
    label = measured_label(candidates, rows, trial_id=trial, decision_sha256=decision)
    # Keep ambiguous and all-failure trials for coverage without teaching a fabricated action.
    with Path(output).open('xb') as f:
        np.savez_compressed(f, trial_id=trial, decision_sha256=decision, candidate_ids=candidates,
                            tokens=tokens, query_label=np.int64(-1 if label is None else label))


def query_batch(paths, candidate_ids):
    sequences, targets = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            if sample['candidate_ids'].tolist() != list(candidate_ids):
                raise ValueError('Different actuator or candidate order in student batch')
            target = sample['query_label'].item()
            if target == -1:
                continue
            if type(target) is not int or not 0 <= target < len(candidate_ids):
                raise ValueError('Invalid measured query class')
            sequences.append(torch.from_numpy(sample['tokens']))
            targets.append(target)
    if not targets:
        raise ValueError('No uniquely measured successful query in this batch')
    lengths = torch.tensor([len(s) for s in sequences]); tokens = nn.utils.rnn.pad_sequence(sequences)
    return tokens, torch.arange(len(tokens))[:, None] < lengths, torch.tensor(targets)


class ConstructedQueryStudent(nn.Module):
    def __init__(self, token_dim, candidates, *, width=128, context_length=300, layers=2, heads=4):
        super().__init__()
        self.candidates = tuple(candidates)
        if len(set(self.candidates)) != len(self.candidates) or len(self.candidates) < 2:
            raise ValueError('Require distinct actuator candidates')
        self.core = CausalDecoderCore(token_dim, width=width, memory_length=context_length, layers=layers, heads=heads)
        self.readout = nn.Linear(width, len(self.candidates))

    def forward(self, tokens, valid):
        features = self.core.sequence(tokens, reset=torch.zeros_like(valid), valid=valid)
        last = valid.long().sum(0) - 1
        return self.readout(features[last, torch.arange(len(last), device=tokens.device)])

    def query_loss(self, paths):
        tokens, valid, target = query_batch(paths, self.candidates)
        device = self.readout.weight.device
        return F.cross_entropy(self(tokens.to(self.readout.weight), valid.to(device)), target.to(device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public', 'evidence', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    export_query(args.public, args.evidence, args.output)
