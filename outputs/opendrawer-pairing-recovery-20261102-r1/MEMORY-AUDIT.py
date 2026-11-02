"""Audit actual frozen candidates and saved support rows on allocated Linux CPU."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
from fluffy_bassoon.meta_rl.vla_history import load_policy

assert sys.platform == 'linux' and not torch.cuda.is_available()
torch.set_num_threads(2)
ROOT = Path('/mowenjie/fluffy-bassoon-2')
CORE = ROOT / 'worktrees/opendrawer-ad-dpt-20261102-r1'
PIN = '68f2c5aa33218709453a7821f6c6223ac3803035'
assert subprocess.check_output(['git', '-C', str(CORE), 'rev-parse', 'HEAD'], text=True).strip() == PIN
TRAIN = ROOT / 'outputs/opendrawer-ppo-ad-dpt-20261102-r1'
OUT = ROOT / 'outputs/opendrawer-pairing-recovery-20261102-r1'
report = dict(checked_at=datetime.now(timezone.utc).isoformat(), source=PIN,
              device='allocated Linux CPU', checks=[], simulated_5shot_outcomes=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def replay(actor, rows, reward_override=None):
    for index, row in enumerate(rows):
        actor.controller.act(row['features'], deterministic=True)
        reward = (reward_override if reward_override is not None and index == len(rows)-1 else row['reward'])
        actor.controller.observe(row['control'], reward, row['done'])


for family in ('ad', 'dpt'):
    full, _ = load_policy(TRAIN / 'fits' / f'{family}_full/candidate.pt',
                          expected_sha256=sha(TRAIN / 'fits' / f'{family}_full/candidate.pt'), device='cpu')
    static, _ = load_policy(TRAIN / 'fits' / f'{family}_static/candidate.pt',
                            expected_sha256=sha(TRAIN / 'fits' / f'{family}_static/candidate.pt'), device='cpu')
    frozen = {id(p): {k: v.clone() for k, v in p.state_dict().items()} for p in (full, static)}
    phase = ROOT / 'outputs' / ('opendrawer-ad-first-id-ood-20261102-r2' if family == 'ad'
                                else 'opendrawer-dpt-full-query-recovery-20261102-r1')
    for cohort in ('ID', 'OOD'):
        case = phase / 'episodes' / cohort / 'case-000' / f'{family}_full'
        support_row = json.loads((case / 'K1/support/result.json').read_text())
        query_row = json.loads((case / 'K1/query/result.json').read_text())
        assert sha(support_row['records']['path']) == support_row['records']['sha256']
        assert sha(query_row['records']['path']) == query_row['records']['sha256']
        support = torch.load(support_row['records']['path'], map_location='cpu', weights_only=False)
        query = torch.load(query_row['records']['path'], map_location='cpu', weights_only=False)
        assert support[-1]['reward'] == support_row['success'] and support[-1]['done']
        features = np.array(query[0]['features'], copy=True)
        tensor = torch.as_tensor(features[None])

        # Production creates a fresh controller for each full-model case, while sharing immutable weights.
        one_shot = HistoryResidualActor(full, mode='full', reset_each_episode=False)
        replay(one_shot, support)
        before = deepcopy(one_shot.controller.state)
        assert int(before['length'][0]) == len(support)
        assert one_shot.controller.previous_reward == support_row['success']
        assert one_shot.controller.previous_done and one_shot.controller.episode_start
        one_shot.begin_episode()
        assert state_equal(before, one_shot.controller.state), 'Query begin erased support memory'

        # A second actor's cache and reset must not mutate the first actor's cache.
        other = HistoryResidualActor(full, mode='full')
        other.act(tensor, deterministic=True)
        assert state_equal(before, one_shot.controller.state), 'Shared weights share mutable controller memory'
        assert other.controller.state['memory'].data_ptr() != one_shot.controller.state['memory'].data_ptr()

        captured = []
        step = full.step
        def observed_step(token, state=None, **kwargs):
            captured.append((token.detach().clone(), deepcopy(state)))
            return step(token, state, **kwargs)
        full.step = observed_step
        try:
            one_shot.act(tensor, deterministic=True)
        finally:
            full.step = step
        token, prior_state = captured[0]
        assert int(prior_state['length'][0]) == len(support)
        assert np.array_equal(token[0, full.config.observation_dim:full.config.observation_dim+6].numpy(), support[-1]['control'])
        assert token[0, -3:].tolist() == [float(support_row['success']), 1., 1.]

        # The production K0 actor resets at every case, including all feedback fields.
        k0 = HistoryResidualActor(full, mode='full')
        replay(k0, support)
        k0.begin_episode()
        assert k0.controller.state is None and k0.controller.previous_reward == 0.
        assert not k0.controller.previous_done and np.count_nonzero(k0.controller.previous_noise) == 0
        actual, _, _ = k0.act(tensor, deterministic=True)
        fresh = HistoryResidualActor(full, mode='full')
        expected, _, _ = fresh.act(tensor, deterministic=True)
        assert torch.equal(actual, expected) and state_equal(k0.controller.state, fresh.controller.state)

        # Static ignores previous observations, actions and outcomes even without an episode reset.
        blind = HistoryResidualActor(static, mode='static', reset_each_episode=False)
        replay(blind, support)
        actual, _, _ = blind.act(tensor, deterministic=True)
        empty = HistoryResidualActor(static, mode='static')
        expected, _, _ = empty.act(tensor, deterministic=True)
        assert torch.equal(actual, expected)

        report['checks'].append(dict(family=family, cohort=cohort, case=0,
            support_decisions=len(support), support_terminal_feedback=support_row['success'],
            query_prior_cache_length=int(prior_state['length'][0]),
            support_preserved_at_query=True, query_token_has_previous_actual_terminal_feedback=True,
            actor_caches_isolated=True, K0_reset_matches_fresh_actor_exactly=True,
            static_history_blind_exactly=True, policy_weights_and_buffers_frozen=True,
            context_capacity=full.config.context_length,
            five_support_plus_query_decisions=6*len(support),
            five_shot_fits_context=6*len(support) <= full.config.context_length,
            training_window_limit=64, five_shot_exceeds_training_window=6*len(support) > 64))
    for p in (full, static):
        assert all(torch.equal(value, frozen[id(p)][key]) for key, value in p.state_dict().items())
    print(json.dumps(dict(family=family, passed=True)), flush=True)
report['passed'] = True
report['scope'] = ('Actual selected AD/DPT full/static checkpoints; retained ID/OOD case0 support/query feature rows. '
                   'Structural memory/feedback isolation on CPU; not a new simulator outcome or cross-device action parity claim.')
(OUT / 'MEMORY-AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
