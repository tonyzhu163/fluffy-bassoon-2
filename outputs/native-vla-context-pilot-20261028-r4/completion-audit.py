"""Read saved artifacts on the existing allocated Baige CPU machine; no model execution."""
import hashlib
import json
from pathlib import Path
import socket

import torch


assert socket.gethostname() == 'aibox-r855e239307b-6bfb688cdd-dmr8f'
torch.set_num_threads(2)
root = Path('/mowenjie/fluffy-bassoon-2/outputs/native-vla-context-pilot-20261028-r4')
run = root/'execution'
expected_source = '0d6dee3d2a792dbede86fe754425bdd361e3cdde'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_history(path):
    trial = torch.load(path, map_location='cpu', weights_only=False)
    records, rewards = trial['records'], trial['rewards']
    assert trial['format'] == 'native_pi05_prefix_inputs_history_v2'
    assert len(records) == 64 and rewards.shape == (64,1)
    assert trial['simulator_steps'] == 320
    previous = torch.zeros(1,7)
    for index, record in enumerate(records):
        feedback = record['feedback']
        assert feedback.shape == (1,11) and feedback.dtype == torch.float32
        assert torch.isfinite(feedback).all()
        torch.testing.assert_close(feedback[:,:7], previous)
        assert feedback[0,7] == (rewards[index-1,0] if index else 0)
        assert feedback[0,8] == int(index>0 and index%16 == 0)
        torch.testing.assert_close(feedback[0,9:], torch.tensor([index//16/4,index%16/16]))
        command = record['requested_commands']
        assert command.shape == (1,5,7) and torch.isfinite(command).all()
        assert all(torch.isfinite(t).all() for t in record['prefix_inputs'])
        previous = command[:,-1]
        if index%16 != 15:
            assert rewards[index,0] == 0
    assert rewards[15::16,0].tolist() == trial['successes']
    return dict(successes=trial['successes'],rewards=float(rewards.sum()),
                chunks=len(records),sha256=sha(path)),trial


checkpoint = torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
assert checkpoint['source_commit'] == expected_source
assert checkpoint['complete_updates'] == 20 and checkpoint['ppo_done'] is True
assert all(torch.isfinite(t).all() for t in checkpoint['weights'].values())
assert checkpoint['optimizer']['state'] and checkpoint['rng']
assert checkpoint['sampler'] and checkpoint['generator'].numel()>0
histories = []
for index, (path,digest) in enumerate(zip(checkpoint['trial_paths'],checkpoint['trial_hashes'],strict=True)):
    assert Path(path) == run/f'history-{index}.pt'
    result,_ = inspect_history(Path(path))
    assert result['sha256'] == digest
    histories.append(result)
assert len(histories) == 5
ppo, rollout = inspect_history(run/'ppo-rollout.pt')
old_values = torch.stack([r['stochastic']['old_value'] for r in rollout['records']])
assert ppo['rewards'] == 0 and old_values.eq(0).all()
assert all(torch.isfinite(r['stochastic']['old_log_prob']).all() for r in rollout['records'])
summary = json.loads((run/'summary.json').read_text())
assert summary['status'] == 'completed' and summary['warmstart_updates'] == 20
evaluation = {}
for lesion in ('normal','memory_wiped','reward_blind'):
    raw = [json.loads((run/f'evaluation-{lesion}-{index}.json').read_text()) for index in (0,1)]
    assert [t['theta'] for t in raw] == [-30.,30.]
    assert [t['successes'] for t in raw] == summary['evaluation'][lesion]
    evaluation[lesion] = dict(successes=sum(sum(t['successes']) for t in raw),attempts=8,
                             per_attempt=[sum(t['successes'][i] for t in raw) for i in range(4)])
smoke = json.loads((run/'ppo-smoke.json').read_text())
assert smoke['gradient_norm'] == 0 and smoke['loss'] == 0 and smoke['max_replay_error'] == 0
result = dict(status='audited',source_commit=expected_source,checkpoint_sha256=sha(run/'latest.pt'),
              checkpoint_updates=checkpoint['complete_updates'],checkpoint_ppo_done=True,
              optimizer_states=len(checkpoint['optimizer']['state']),
              optimizer_step_counts=sorted({int(s['step']) for s in checkpoint['optimizer']['state'].values()}),
              rng_fields=sorted(checkpoint['rng']),sampler_and_generator_preserved=True,
              trainable_weights_finite=True,history_feedback_and_reward_checks=True,
              histories=histories,ppo_rollout=ppo,ppo_old_values_all_zero=True,
              ppo_gradient_norm=0.,evaluation=evaluation,task_simulator_steps=13*320,
              interpretation='PPO rollout had zero rewards/values/advantages; zero PPO gradient is not RL learning. '
              'Adam momentum from warm-start can still move parameters on that optimizer step. '
              'Tiny evaluation has no paired frozen-base or warmstart-only evaluation control.')
with (root/'completion-audit.json').open('x') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))
