"""Explicit fixed-C10 confirmation on the sealed, policy-free fresh scene bank."""
import json
from pathlib import Path
from types import MethodType

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'MOTUS-FRESH-C10-CONFIRMATION-PROTOCOL.json'
BANK = 'MOTUS-FRESH-CONFIRMATION-BANK.json'
FRESH_FIELDS = ('fresh_confirmation_readout', 'fresh_confirmation_bank_sha256',
    'fresh_confirmation_protocol_sha256')


def fresh_mode(plan):
    enabled = plan.get('fresh_confirmation_readout', False)
    assert type(enabled) is bool
    if enabled:
        assert plan['extended_source_curve_readout'] is True
        assert plan['algorithm'] == 'grpo' and plan['bank'] == 'fresh_confirmation'
        assert plan['selected_training_iteration'] == 10
    else:
        assert plan.get('bank') != 'fresh_confirmation', 'Fresh scenes require the explicit fixed-C10 route'
    return enabled


def qualified_bank(plan):
    """Bind local manifest and raw independent qualification before policy use."""
    assert fresh_mode(plan)
    protocol = json.loads((ROOT/PROTOCOL).read_bytes())
    assert sha(ROOT/PROTOCOL) == plan['fresh_confirmation_protocol_sha256']
    assert protocol['bank_manifest'] == BANK
    assert protocol['endpoint_training_iteration'] == 10
    assert protocol['minimum_success_gain_percentage_points'] == 10
    assert protocol['maximum_exclusive_p_value'] == .05 and protocol['single_use']
    assert protocol['native_rollout_batch'] == 2 and protocol['total_policy_episodes'] == 128
    bank = json.loads((ROOT/BANK).read_bytes())
    assert sha(ROOT/BANK) == plan['fresh_confirmation_bank_sha256'] == protocol['bank_sha256']
    assert bank['format'] == 'motus_fresh_confirmation_bank_v1'
    assert bank['task'] == protocol['task'] == 'place_phone_stand'
    assert bank['target_training_iteration'] == 10
    assert not bank['policy_outcomes_observed'] and bank['original_sampler_inventory_preserved']
    seeds = bank['seeds']
    assert len(seeds) == len(set(seeds)) == 64 and all(type(s) is int for s in seeds)
    from motus_fresh_bank import declaration, readback
    scan, _ = declaration()
    excluded = json.loads((ROOT/scan['excluded_train_file']).read_bytes())['success_seeds']
    assert not set(seeds) & set(excluded+scan['excluded_native_eval_seeds'])
    raw = Path(protocol['qualification_result'])
    assert sha(raw) == protocol['qualification_result_sha256'] == bank['qualification_result_sha256']
    verified = readback(raw.parent)
    assert verified['seeds'] == seeds and verified['bank_sha256'] == protocol['bank_sha256']
    audit_path = Path(protocol['independent_bank_audit'])
    assert sha(audit_path) == protocol['independent_bank_audit_sha256']
    audit = json.loads(audit_path.read_bytes())
    assert audit['status'] == 'passed_independent_native_curobo_fresh_confirmation_bank'
    assert audit['seeds'] == seeds and audit['bank_sha256'] == protocol['bank_sha256']
    assert audit['model_calls'] == audit['optimizer_updates'] == 0 and audit['cleanup_complete']
    assert audit['native_collision_configuration_unchanged'] and audit['native_success_seed_inventory_unchanged']
    # Fresh images cannot equal old DEV images. Bind the separately verified
    # original-SFT calibration rather than weakening or fabricating that check.
    calibration = json.loads((ROOT/'MOTUS-SOURCE-GRPO-DEV02-READBACK.json').read_bytes())
    assert sha(calibration['result']) == calibration['sha256']
    assert calibration['first48_SFT_predictions_pixels_actions_RNG_exact']
    assert calibration['full24_worker_restore_each_policy']
    return seeds


def readout_bank(plan):
    if fresh_mode(plan):
        return qualified_bank(plan)
    assert plan['bank'] in ('dev', 'confirmation')
    return json.loads((ROOT/'MOTUS-SOURCE-READOUT-CASES.json').read_bytes())[plan['bank']]


def result_bank(result):
    if fresh_mode(result):
        assert result['new_confirmation'] is True and result['fixed_DEV_only'] is False
        return qualified_bank(result)
    return readout_bank(result)


def install_readout_reset(owner, evaluation):
    """Use explicit physical seeds without replacing the restored sampler inventory."""
    import torch
    original = evaluation.reset

    def reset(env, env_idx=None, env_seeds=None):
        indices = list(range(env.num_envs)) if env_idx is None else list(env_idx)
        seeds = [owner._readout_selected[int(i)] for i in indices]
        env.reset_state_ids[indices] = torch.tensor(seeds, dtype=torch.long,
            device=env.reset_state_ids.device)
        result = original(env_idx=env_idx, env_seeds=seeds)
        for index, seed in zip(indices, seeds, strict=True):
            sub = env.venv.envs[int(index)]
            assert int(sub.env_seed) == int(sub.task.ep_num) == seed, 'Readout reset substituted a physical seed'
        return result

    evaluation.reset = MethodType(reset, evaluation)
