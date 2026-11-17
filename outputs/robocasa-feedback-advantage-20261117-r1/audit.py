"""Independent raw-record verification of Stage 1 and Stage 2 results and sampler RNGs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_and_eval import (
    ACTIONS,
    METHODS,
    SEEDS,
    HistoryActor,
    HistoryQCritic,
    HistoryVCritic,
    arm_name,
    binary_auc,
    build_drift_trials,
    build_probed_trials,
    evaluate_stage1,
    importance_weights,
    make_sequence_tensors,
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fsum_mean(xs):
    return math.fsum(xs) / len(xs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    summary = json.loads((args.panel / 'SUMMARY.json').read_text())
    result = json.loads((args.fit / 'RESULT.json').read_text())
    events = [json.loads(line) for line in (args.panel / 'events.jsonl').read_text().splitlines() if line.strip()]
    event_episodes = [e for e in events if e.get('kind') == 'episode']
    assert len(event_episodes) == len(summary['records']), (len(event_episodes), len(summary['records']))
    for e_row, s_row in zip(event_episodes, summary['records'], strict=True):
        assert e_row['scene'] == s_row['scene'] and e_row['arm'] == s_row['arm']
        assert e_row['success'] == s_row['success'] and e_row['terminal_sha256'] == s_row['terminal_sha256']

    admitted = summary['admitted_scenes']
    assert len(admitted) == 16
    assert result['train_scenes'] == admitted[:8]
    assert result['eval_scenes'] == admitted[8:16]
    assert result['panel_summary_sha256'] == sha256_file(args.panel / 'SUMMARY.json')

    by_key = {(r['scene'], r['arm']): r for r in summary['records']}
    checks = 0
    for scene in admitted:
        nom = by_key[(scene, 'nominal')]
        rep = by_key[(scene, 'repeat_tape')]
        assert nom['success'] and nom['decision'] is not None
        assert rep['success'] == nom['success'] and rep['controls'] == nom['controls']
        assert rep['terminal_sha256'] == nom['terminal_sha256']
        assert rep['decision_observation_sha256'] == nom['decision_observation_sha256']
        assert rep['matched_pre_action_states'] == nom['controls']
        checks += 4
        assert (scene, 'repeat_closed_loop') in by_key
        for sign in (-1, 1):
            for u in ACTIONS:
                b = by_key[(scene, arm_name(sign, u))]
                assert b['matched_pre_action_states'] == nom['decision'] + 1
                assert b['decision_observation_sha256'] == nom['decision_observation_sha256']
                checks += 2

    # 1. Recompute Stage 1 completely independently and verify exact match.
    re_stage1 = evaluate_stage1(result['train_scenes'], result['eval_scenes'], by_key)
    assert re_stage1['best_fixed_u_on_train'] == result['stage1']['best_fixed_u_on_train']
    for panel_key in ('probed_train_cross', 'probed_eval_cross', 'probed_eval_same', 'probed_eval_cross_all',
                      'drift_train_cross', 'drift_eval_cross', 'drift_eval_same'):
        for arm_key, reported in result['stage1'][panel_key].items():
            if not isinstance(reported, dict) or 'attempt2_success' not in reported:
                continue
            re_arm = re_stage1[panel_key][arm_key]
            for metric in ('action_accuracy', 'attempt2_success', 'cumulative_success'):
                assert abs(reported[metric]['point'] - re_arm[metric]['point']) <= 1e-12
                checks += 1

    # 2. Recompute Stage 2 step-1024 predictions, critic metrics, and sampler RNGs from saved checkpoints.
    checkpoint_hashes = {}
    for suite_name in ('probed_action_reward', 'uncompensated_drift'):
        if suite_name == 'probed_action_reward':
            train_base = build_probed_trials(result['train_scenes'], by_key, mode='all_ordered')
            eval_cross = build_probed_trials(result['eval_scenes'], by_key, mode='cross_scene')
            eval_same = build_probed_trials(result['eval_scenes'], by_key, mode='same_scene')
        else:
            train_base = build_drift_trials(result['train_scenes'], by_key, mode='all_ordered')
            eval_cross = build_drift_trials(result['eval_scenes'], by_key, mode='cross_scene')
            eval_same = build_drift_trials(result['eval_scenes'], by_key, mode='same_scene')
        n_train = len(train_base) * len(ACTIONS)
        eval_behavior = [dict(b, u2=u2, a2_idx=ai, r2=float(b['outcomes'][u2]))
                         for b in eval_cross for ai, u2 in enumerate(ACTIONS)]

        for seed in SEEDS:
            seed_data = result['stage2'][suite_name]['seeds'][str(seed)]
            for m_idx, method in enumerate(METHODS):
                ckpt_path = args.fit / suite_name / f'seed-{seed}' / f'{method}-step-1024.pt'
                checkpoint_hashes[f'{suite_name}/seed-{seed}/{method}'] = sha256_file(ckpt_path)
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                assert ckpt['step'] == 1024 and ckpt['seed'] == seed and ckpt['method'] == method

                # Reconstruct exact sampler RNG state.
                sampler = np.random.default_rng(10000 * seed + 100 + m_idx)
                seen = np.zeros(n_train, dtype=np.int32)
                for _ in range(1024):
                    seen[sampler.integers(n_train, size=32)] += 1
                assert sampler.bit_generator.state == ckpt['rng']['sampler']
                assert seen.tolist() == ckpt['seen_counts']
                assert (seen > 0).all(), f'Incomplete query coverage in {suite_name}/{seed}/{method}'
                assert int(seen.sum()) == 1024 * 32
                checks += 4

                norm_stats = {k: np.asarray(v, dtype=np.float32) for k, v in ckpt['norm_stats'].items()}
                actor = HistoryActor(result['stage2'][suite_name]['input_dim'])
                actor.load_state_dict(ckpt['model'], strict=True)
                actor.eval()

                rep_ckpt = seed_data['methods'][method]['checkpoints']['1024']
                assert rep_ckpt['unique_queries_covered'] == n_train
                for panel_name, trials in (('eval_cross', eval_cross), ('eval_same', eval_same)):
                    for var in ('own', 'matched_donor', 'cleared', 'feedback_masked', 'counterfactual_reward'):
                        use_var = 'cleared' if method == 'memoryless' else var
                        vt, vr, vv = make_sequence_tensors(trials, norm_stats, suite=suite_name, variant=use_var)
                        with torch.no_grad():
                            pred_idx = actor.sequence_logits(vt, vr, vv)[-1].argmax(-1).tolist()
                        preds = [ACTIONS[k] for k in pred_idx]
                        re_att2 = fsum_mean([float(t['outcomes'][u]) for t, u in zip(trials, preds)])
                        re_cum = fsum_mean([1.0 if (t['r_sup'] > 0.5 or t['outcomes'][u] > 0.5) else 0.0
                                            for t, u in zip(trials, preds)])
                        rep_var = rep_ckpt['eval'][panel_name][var]
                        assert abs(re_att2 - rep_var['attempt2_success']['point']) <= 1e-12
                        assert abs(re_cum - rep_var['cumulative_success']['point']) <= 1e-12
                        checks += 2

                # Check critic metrics on first method (same critic across methods in a seed).
                if m_idx == 0:
                    dim = result['stage2'][suite_name]['input_dim']
                    q_net = HistoryQCritic(dim)
                    v_net = HistoryVCritic(dim)
                    q_net.load_state_dict(ckpt['critic_q'], strict=True)
                    v_net.load_state_dict(ckpt['critic_v'], strict=True)
                    q_net.eval()
                    v_net.eval()
                    vt, vr, vv = make_sequence_tensors(eval_behavior, norm_stats, suite=suite_name, variant='own')
                    ev_a2 = torch.tensor([x['a2_idx'] for x in eval_behavior], dtype=torch.long)
                    ev_r2 = np.array([x['r2'] for x in eval_behavior], dtype=np.float64)
                    with torch.no_grad():
                        ev_q = q_net.forward_query(vt, vr, vv, ev_a2).numpy().astype(np.float64)
                        ev_v = v_net.forward_query(vt, vr, vv).numpy().astype(np.float64)
                    re_auc = binary_auc(ev_q - ev_v, ev_r2)
                    re_mse = fsum_mean(((ev_q - ev_r2) ** 2).tolist())
                    assert abs(re_auc - seed_data['critic']['eval_auc']) <= 1e-6
                    assert abs(re_mse - seed_data['critic']['eval_q_mse']) <= 1e-6
                    ev_w1 = importance_weights(torch.from_numpy(ev_q - ev_v), 1.0).numpy()
                    re_ess1 = math.fsum(ev_w1.tolist()) ** 2 / (len(ev_w1) * math.fsum((ev_w1 ** 2).tolist()))
                    assert abs(re_ess1 - seed_data['critic']['eval_weights']['dit_tau1']['ess_fraction']) <= 1e-6
                    checks += 3

    audit = dict(
        passed=True,
        verified_checks=checks,
        admitted_scenes=admitted,
        screened_scenes=summary['screened_count'],
        panel_episodes=summary['episodes'],
        panel_summary_sha256=result['panel_summary_sha256'],
        fit_result_sha256=sha256_file(args.fit / 'RESULT.json'),
        checkpoint_sha256s=checkpoint_hashes,
    )
    write_json(args.output, audit)
    print(json.dumps(dict(passed=True, verified_checks=checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
