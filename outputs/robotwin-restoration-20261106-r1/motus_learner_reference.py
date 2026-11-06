"""Independent NumPy reference for the actual native chunk-level learner batch.

Run only on allocated Linux. Native rewards already include its timeout
bootstrap; this audits the saved native recurrence, not a new timeout policy.
"""
import numpy as np


def audit(batch, algorithm):
    def array(key):
        return batch[key].detach().cpu().numpy()
    reward = array('rewards').astype(np.float64).sum(axis=-1)
    done = array('dones').astype(bool).any(axis=-1)
    mask = array('loss_mask').astype(bool).any(axis=-1)
    steps, lanes = reward.shape
    assert done.shape == (steps+1, lanes) and mask.shape == reward.shape
    report = dict(algorithm=algorithm,chunk_steps=steps,lanes=lanes,
        valid_native_decisions=int(mask.sum()),
        independent_reference='NumPy float64; no native algorithm/preprocessing helpers',
        rewards_already_contain_native_timeout_bootstrap=True)
    if algorithm == 'ppo':
        value = array('prev_values').astype(np.float64).reshape(steps+1, lanes)
        carry = np.zeros(lanes, dtype=np.float64)
        returns = np.empty_like(reward)
        for time in reversed(range(steps)):
            continuation = ~done[time+1]
            delta = reward[time] + .99 * value[time+1] * continuation - value[time]
            carry = delta + .99 * .95 * continuation * carry
            returns[time] = carry + value[time]
        advantages = returns - value[:-1]
        valid = advantages[mask]
        assert valid.size > 1
        advantages = (advantages-valid.mean()) / (valid.std(ddof=1)+1e-5)
        errors = dict(advantages=float(np.max(np.abs(array('advantages')[...,0]-advantages))),
                      returns=float(np.max(np.abs(array('returns')[...,0]-returns))))
    else:
        assert algorithm == 'grpo' and lanes % 8 == 0 and 'returns' not in batch
        score = np.zeros(lanes, dtype=np.float64)
        for time in reversed(range(steps)):
            score = score * ~done[time+1] + reward[time]
        groups = score.reshape(-1, 8)
        std = groups.std(axis=1, ddof=1, keepdims=True)
        group_advantage = ((groups-groups.mean(axis=1,keepdims=True))/(std+1e-6)).reshape(lanes)
        advantages = group_advantage[None,:] * mask
        errors = dict(advantages=float(np.max(np.abs(array('advantages')[...,0]-advantages))))
        report.update(tied_groups=int((std == 0).sum()),groups=int(len(groups)),
                      minimum_score=float(score.min()),maximum_score=float(score.max()),
                      all_tied_advantages_exact_zero=bool((advantages[:,np.repeat(std[:,0] == 0,8)] == 0).all()))
    report.update(max_abs_error=errors,tolerance=1e-5,
        passed=all(np.isfinite(error) and error <= 1e-5 for error in errors.values()))
    return report


if __name__ == '__main__':
    import json
    import os
    import sys
    import torch
    import rlinf.algorithms
    from rlinf.algorithms.registry import calculate_adv_and_returns
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    steps, lanes, chunk = 6, 16, 32
    rewards = torch.zeros(steps,lanes,chunk)
    dones = torch.zeros(steps+1,lanes,chunk,dtype=torch.bool)
    mask = torch.ones_like(rewards,dtype=torch.bool)
    for lane in range(lanes):
        if lane < 8 and lane % 3 == 0:
            rewards[1,lane,-1] = 1
            dones[2,lane,:] = True
            mask[2:,lane,:] = False
    dones[-1,:,:] = True
    values = torch.linspace(-.3,.4,(steps+1)*lanes).reshape(steps+1,lanes,1)
    rows = []
    for algorithm in ('ppo','grpo'):
        batch = dict(rewards=rewards.clone(),dones=dones.clone(),loss_mask=mask.clone())
        if algorithm == 'ppo': batch['prev_values'] = values.clone()
        actual = calculate_adv_and_returns(task_type='embodied',adv_type='gae' if algorithm == 'ppo' else 'grpo',
            rewards=batch['rewards'],dones=batch['dones'],values=batch.get('prev_values'),
            loss_mask=batch['loss_mask'],loss_mask_sum=None,reward_type='chunk_level',
            num_action_chunks=chunk,gamma=.99,gae_lambda=.95,group_size=8)
        batch.update(actual)
        row = audit(batch,algorithm)
        assert row['passed'], row
        rows.append(row)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='complete',fixtures=rows,cuda_initialized=False)))
