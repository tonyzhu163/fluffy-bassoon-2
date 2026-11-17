"""Fit Stage 1 reference selectors and Stage 2 causal-history / DIT students on OpenDrawer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[2]
if str(CODE / 'src') not in sys.path:
    sys.path.insert(0, str(CODE / 'src'))

from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore

ACTIONS = (-1, 0, 1)
MILESTONES = (64, 128, 256, 512, 1024)
SEEDS = (0, 1, 2)
METHODS = ('dpt', 'uniform_bc', 'dit_tau1', 'dit_tau01', 'known_outcome_tau01', 'memoryless')


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


def arm_name(sign, u):
    return f'b{sign:+d}-u{u:+d}'


def bootstrap_ci(scene_num_a, scene_den_a, scene_num_b=None, scene_den_b=None, *, seed=20261117, draws=10000):
    rng = np.random.default_rng(seed)
    n_scenes = len(scene_num_a)
    counts = rng.multinomial(n_scenes, np.full(n_scenes, 1.0 / n_scenes), size=draws)
    a_num = counts @ np.asarray(scene_num_a, dtype=np.float64)
    a_den = counts @ np.asarray(scene_den_a, dtype=np.float64)
    rate_a = np.divide(a_num, a_den, out=np.full(draws, np.nan), where=a_den > 0)
    if scene_num_b is None:
        point = float(np.sum(scene_num_a) / np.sum(scene_den_a))
        lo, hi = np.nanquantile(rate_a, [0.025, 0.975]).tolist()
        return dict(point=point, ci95=[float(lo), float(hi)])
    b_num = counts @ np.asarray(scene_num_b, dtype=np.float64)
    b_den = counts @ np.asarray(scene_den_b, dtype=np.float64)
    rate_b = np.divide(b_num, b_den, out=np.full(draws, np.nan), where=b_den > 0)
    diff = rate_a - rate_b
    point = float(np.sum(scene_num_a) / np.sum(scene_den_a) - np.sum(scene_num_b) / np.sum(scene_den_b))
    lo, hi = np.nanquantile(diff, [0.025, 0.975]).tolist()
    return dict(point=point, ci95=[float(lo), float(hi)])


def summarize_trials(trials, scenes, predictions, baseline_predictions=None):
    """Aggregate action accuracy, Attempt-2 success, and cumulative (1-or-2) success with scene CIs."""
    scene_index = {s: i for i, s in enumerate(scenes)}
    n = len(scenes)
    acc_num, att2_num, cum_num, den = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    b_att2 = np.zeros(n)
    b_cum = np.zeros(n)
    wins = losses = ties = 0
    for idx, (trial, pred) in enumerate(zip(trials, predictions, strict=True)):
        i = scene_index[trial['query_scene']]
        u = int(pred)
        assert u in ACTIONS
        r2 = float(trial['outcomes'][u])
        r1 = float(trial['r_sup'])
        cum = 1.0 if (r1 > 0.5 or r2 > 0.5) else 0.0
        acc_num[i] += float(u == -trial['z'])
        att2_num[i] += r2
        cum_num[i] += cum
        den[i] += 1.0
        if baseline_predictions is not None:
            bu = int(baseline_predictions[idx])
            br2 = float(trial['outcomes'][bu])
            bcum = 1.0 if (r1 > 0.5 or br2 > 0.5) else 0.0
            b_att2[i] += br2
            b_cum[i] += bcum
            if r2 > br2:
                wins += 1
            elif r2 < br2:
                losses += 1
            else:
                ties += 1
    out = dict(
        trials=len(trials),
        action_accuracy=bootstrap_ci(acc_num, den),
        attempt2_success=bootstrap_ci(att2_num, den),
        cumulative_success=bootstrap_ci(cum_num, den),
    )
    if baseline_predictions is not None:
        out['diff_attempt2'] = bootstrap_ci(att2_num, den, b_att2, den)
        out['diff_cumulative'] = bootstrap_ci(cum_num, den, b_cum, den)
        out['paired_attempt2_counts'] = dict(wins=wins, losses=losses, ties=ties)
    return out


def fit_ridge(X, y, lam=1e-2):
    """Fit centered ridge regression on training split only."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0).clip(1e-6)
    Xn = (X - mean) / std
    y_mean = float(y.mean())
    yc = y - y_mean
    w = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ yc)
    return dict(mean=mean, std=std, y_mean=y_mean, w=w)


def predict_ridge(model, X):
    Xn = (np.asarray(X, dtype=np.float64) - model['mean']) / model['std']
    return model['y_mean'] + Xn @ model['w']


def build_panel_records(summary):
    admitted = summary['admitted_scenes']
    if len(admitted) != 16:
        raise ValueError(f'Expected 16 admitted scenes, got {len(admitted)}')
    by_key = {}
    for r in summary['records']:
        by_key[(r['scene'], r['arm'])] = r
    repeat_checks = []
    for s in admitted:
        nom = by_key[(s, 'nominal')]
        rep = by_key[(s, 'repeat_tape')]
        cl = by_key[(s, 'repeat_closed_loop')]
        assert rep['success'] == nom['success'] and rep['controls'] == nom['controls']
        assert rep['terminal_sha256'] == nom['terminal_sha256']
        assert rep['decision_observation_sha256'] == nom['decision_observation_sha256']
        repeat_checks.append(dict(
            scene=s,
            nominal_success=bool(nom['success']),
            nominal_controls=int(nom['controls']),
            decision=int(nom['decision']),
            first_grasp=nom['first_grasp'],
            repeat_tape_exact=True,
            repeat_cl_success_equal=bool(cl['success'] == nom['success']),
            repeat_cl_terminal_exact=bool(cl['terminal_sha256'] == nom['terminal_sha256']),
            repeat_cl_first_state_diff=cl['first_state_diff'],
            repeat_cl_first_command_diff=cl['first_command_diff'],
            repeat_cl_first_command_max_abs=cl['first_command_max_abs'],
        ))
        for sign in (-1, 1):
            for u in ACTIONS:
                b = by_key[(s, arm_name(sign, u))]
                assert b['matched_pre_action_states'] == nom['decision'] + 1
                assert b['decision_observation_sha256'] == nom['decision_observation_sha256']
    return admitted, by_key, repeat_checks


def two_way_summary(scenes, by_key):
    rows = []
    two_way_scenes = 0
    for s in scenes:
        sign_ok = {}
        for sign in (-1, 1):
            outs = {u: bool(by_key[(s, arm_name(sign, u))]['success']) for u in ACTIONS}
            sign_ok[sign] = bool(outs[-sign] and not outs[0] and not outs[sign])
            rows.append(dict(scene=s, z=sign, outcomes={str(k): v for k, v in outs.items()},
                             two_way_contrast=sign_ok[sign]))
        if sign_ok[-1] and sign_ok[1]:
            two_way_scenes += 1
    return dict(scenes=len(scenes), both_signs_two_way=two_way_scenes, conditions=rows)


def build_probed_trials(scenes, by_key, *, mode):
    """Build two-attempt trials with own and outcome-matched donor supports."""
    trials = []
    for q_idx, q_scene in enumerate(scenes):
        if mode == 'cross_scene':
            sup_indices = [q_idx ^ 1]
            donor_scene_for_q = scenes[q_idx ^ 2]
        elif mode == 'cross_scene_all':
            sup_indices = [j for j in range(len(scenes)) if j != q_idx]
            donor_scene_for_q = None
        elif mode == 'same_scene':
            sup_indices = [q_idx]
            donor_scene_for_q = scenes[q_idx ^ 1]
        elif mode == 'all_ordered':
            sup_indices = list(range(len(scenes)))
            donor_scene_for_q = None
        else:
            raise ValueError(mode)
        q_nom = by_key[(q_scene, 'nominal')]
        q_state = np.asarray(q_nom['decision_state'], dtype=np.float32)
        for s_idx in sup_indices:
            s_scene = scenes[s_idx]
            d_scene = donor_scene_for_q if donor_scene_for_q is not None else scenes[s_idx ^ 1]
            for z in (-1, 1):
                q_outcomes = {u: float(by_key[(q_scene, arm_name(z, u))]['success']) for u in ACTIONS}
                for u_sup in (-1, 1):
                    own_rec = by_key[(s_scene, arm_name(z, u_sup))]
                    r_sup = float(own_rec['success'])
                    # Outcome-matched opposite-sign donor: flip both latent z and support action u_sup.
                    donor_z = -z
                    donor_u_sup = -u_sup
                    donor_rec = by_key[(d_scene, arm_name(donor_z, donor_u_sup))]
                    donor_r_sup = float(donor_rec['success'])
                    trials.append(dict(
                        query_scene=q_scene,
                        support_scene=s_scene,
                        donor_scene=d_scene,
                        z=z,
                        u_sup=u_sup,
                        r_sup=r_sup,
                        support_state=np.asarray(own_rec['decision_state'], dtype=np.float32),
                        query_state=q_state,
                        donor_z=donor_z,
                        donor_u_sup=donor_u_sup,
                        donor_r_sup=donor_r_sup,
                        donor_state=np.asarray(donor_rec['decision_state'], dtype=np.float32),
                        donor_outcome_matched=bool(donor_r_sup == r_sup),
                        outcomes=q_outcomes,
                    ))
    return trials


def build_drift_trials(scenes, by_key, *, mode):
    """Build uncompensated-support (u_sup=0) trials where drift in public proprioception reveals z."""
    trials = []
    for q_idx, q_scene in enumerate(scenes):
        if mode == 'cross_scene':
            sup_indices = [q_idx ^ 1]
            donor_scene_for_q = scenes[q_idx ^ 2]
        elif mode == 'same_scene':
            sup_indices = [q_idx]
            donor_scene_for_q = scenes[q_idx ^ 1]
        elif mode == 'all_ordered':
            sup_indices = list(range(len(scenes)))
            donor_scene_for_q = None
        else:
            raise ValueError(mode)
        q_nom = by_key[(q_scene, 'nominal')]
        q_state = np.asarray(q_nom['decision_state'], dtype=np.float32)
        for s_idx in sup_indices:
            s_scene = scenes[s_idx]
            d_scene = donor_scene_for_q if donor_scene_for_q is not None else scenes[s_idx ^ 1]
            for z in (-1, 1):
                q_outcomes = {u: float(by_key[(q_scene, arm_name(z, u))]['success']) for u in ACTIONS}
                own_rec = by_key[(s_scene, arm_name(z, 0))]
                donor_rec = by_key[(d_scene, arm_name(-z, 0))]
                trials.append(dict(
                    query_scene=q_scene,
                    support_scene=s_scene,
                    donor_scene=d_scene,
                    z=z,
                    u_sup=0,
                    r_sup=float(own_rec['success']),
                    delta_y=float(own_rec['public_delta_eef_y']),
                    cmd_y=float(own_rec['commanded_mean_y']),
                    support_state=np.asarray(own_rec['decision_state'], dtype=np.float32),
                    query_state=q_state,
                    donor_z=-z,
                    donor_u_sup=0,
                    donor_r_sup=float(donor_rec['success']),
                    donor_delta_y=float(donor_rec['public_delta_eef_y']),
                    donor_cmd_y=float(donor_rec['commanded_mean_y']),
                    donor_state=np.asarray(donor_rec['decision_state'], dtype=np.float32),
                    donor_outcome_matched=bool(donor_rec['success'] == own_rec['success']),
                    outcomes=q_outcomes,
                ))
    return trials


def evaluate_stage1(train_scenes, eval_scenes, by_key):
    train_cross = build_probed_trials(train_scenes, by_key, mode='cross_scene')
    eval_cross = build_probed_trials(eval_scenes, by_key, mode='cross_scene')
    eval_same = build_probed_trials(eval_scenes, by_key, mode='same_scene')
    eval_all = build_probed_trials(eval_scenes, by_key, mode='cross_scene_all')

    # Best fixed action on train split.
    best_u = max(ACTIONS, key=lambda u: (
        sum(t['outcomes'][u] for t in train_cross),
        1 if u == 0 else 0,
    ))

    # Current-observation reference on train decision states (symmetric z -> theoretical 0.5 ceiling).
    X_obs_tr = np.stack([t['query_state'] for t in train_cross])
    y_target_tr = np.array([-t['z'] for t in train_cross], dtype=np.float64)
    obs_ridge = fit_ridge(X_obs_tr, y_target_tr, lam=1.0)

    # Feedback reference: bilinear features [u_sup, 2*r_sup - 1, u_sup*(2*r_sup - 1)] predicting optimal u* = -z.
    def fb_feats(trials, *, variant='own'):
        rows = []
        for t in trials:
            if variant == 'own':
                u, r = t['u_sup'], 2.0 * t['r_sup'] - 1.0
            elif variant == 'matched_donor':
                u, r = t['donor_u_sup'], 2.0 * t['donor_r_sup'] - 1.0
            elif variant == 'feedback_masked':
                u, r = t['u_sup'], 0.0
            elif variant == 'counterfactual_reward':
                u, r = t['u_sup'], 1.0 - 2.0 * t['r_sup']
            else:
                raise ValueError(variant)
            rows.append([float(u), float(r), float(u * r)])
        return np.asarray(rows, dtype=np.float64)

    fb_ridge = fit_ridge(fb_feats(train_cross, variant='own'), y_target_tr, lam=1e-2)

    # Uncompensated drift reference (u_sup = 0) predicting u* = -z from public [delta_eef_y, cmd_mean_y].
    train_drift = build_drift_trials(train_scenes, by_key, mode='cross_scene')
    eval_drift_cross = build_drift_trials(eval_scenes, by_key, mode='cross_scene')
    eval_drift_same = build_drift_trials(eval_scenes, by_key, mode='same_scene')
    drift_ridge = fit_ridge(
        np.asarray([[t['delta_y'], t['cmd_y']] for t in train_drift], dtype=np.float64),
        np.array([-t['z'] for t in train_drift], dtype=np.float64),
        lam=1e-2,
    )

    def sign_choice(scores):
        return [1 if float(s) >= 0.0 else -1 for s in scores]

    def score_probed_panel(trials, scenes):
        oracle_p = [-t['z'] for t in trials]
        fixed_p = [best_u for _ in trials]
        uncorr_p = [0 for _ in trials]
        obs_p = sign_choice(predict_ridge(obs_ridge, np.stack([t['query_state'] for t in trials])))
        own_p = sign_choice(predict_ridge(fb_ridge, fb_feats(trials, variant='own')))
        donor_p = sign_choice(predict_ridge(fb_ridge, fb_feats(trials, variant='matched_donor')))
        masked_p = sign_choice(predict_ridge(fb_ridge, fb_feats(trials, variant='feedback_masked')))
        cf_p = sign_choice(predict_ridge(fb_ridge, fb_feats(trials, variant='counterfactual_reward')))
        return dict(
            donor_outcome_matched=sum(t['donor_outcome_matched'] for t in trials),
            oracle=summarize_trials(trials, scenes, oracle_p),
            best_fixed=summarize_trials(trials, scenes, fixed_p),
            uncorrected=summarize_trials(trials, scenes, uncorr_p),
            current_obs_ref=summarize_trials(trials, scenes, obs_p),
            feedback_own=summarize_trials(trials, scenes, own_p, baseline_predictions=fixed_p),
            feedback_own_vs_obs=summarize_trials(trials, scenes, own_p, baseline_predictions=obs_p),
            feedback_own_vs_donor=summarize_trials(trials, scenes, own_p, baseline_predictions=donor_p),
            feedback_own_vs_masked=summarize_trials(trials, scenes, own_p, baseline_predictions=masked_p),
            feedback_matched_donor=summarize_trials(trials, scenes, donor_p),
            feedback_masked=summarize_trials(trials, scenes, masked_p),
            feedback_counterfactual_reward=summarize_trials(trials, scenes, cf_p),
        )

    def score_drift_panel(trials, scenes):
        oracle_p = [-t['z'] for t in trials]
        fixed_p = [best_u for _ in trials]
        own_p = sign_choice(predict_ridge(
            drift_ridge, np.asarray([[t['delta_y'], t['cmd_y']] for t in trials], dtype=np.float64)))
        donor_p = sign_choice(predict_ridge(
            drift_ridge, np.asarray([[t['donor_delta_y'], t['donor_cmd_y']] for t in trials], dtype=np.float64)))
        cleared_p = sign_choice(predict_ridge(drift_ridge, np.zeros((len(trials), 2), dtype=np.float64)))
        return dict(
            oracle=summarize_trials(trials, scenes, oracle_p),
            best_fixed=summarize_trials(trials, scenes, fixed_p),
            drift_own=summarize_trials(trials, scenes, own_p, baseline_predictions=fixed_p),
            drift_own_vs_donor=summarize_trials(trials, scenes, own_p, baseline_predictions=donor_p),
            drift_own_vs_cleared=summarize_trials(trials, scenes, own_p, baseline_predictions=cleared_p),
            drift_matched_donor=summarize_trials(trials, scenes, donor_p),
            drift_cleared=summarize_trials(trials, scenes, cleared_p),
        )

    return dict(
        train_two_way=two_way_summary(train_scenes, by_key),
        eval_two_way=two_way_summary(eval_scenes, by_key),
        best_fixed_u_on_train=int(best_u),
        probed_train_cross=score_probed_panel(train_cross, train_scenes),
        probed_eval_cross=score_probed_panel(eval_cross, eval_scenes),
        probed_eval_same=score_probed_panel(eval_same, eval_scenes),
        probed_eval_cross_all=score_probed_panel(eval_all, eval_scenes),
        drift_train_cross=score_drift_panel(train_drift, train_scenes),
        drift_eval_cross=score_drift_panel(eval_drift_cross, eval_scenes),
        drift_eval_same=score_drift_panel(eval_drift_same, eval_scenes),
    )


class HistoryActor(nn.Module):
    def __init__(self, input_dim, num_actions=3, width=128, layers=2, heads=4, memory_length=4):
        super().__init__()
        self.core = CausalDecoderCore(input_dim, width=width, memory_length=memory_length, layers=layers, heads=heads)
        self.head = nn.Linear(width, num_actions)

    def sequence_logits(self, tokens, reset, valid):
        feat = self.core.sequence(tokens, reset=reset, valid=valid)
        return self.head(feat)


class HistoryQCritic(nn.Module):
    def __init__(self, input_dim, num_actions=3, width=128, layers=2, heads=4, memory_length=4):
        super().__init__()
        self.num_actions = num_actions
        self.core = CausalDecoderCore(input_dim, width=width, memory_length=memory_length, layers=layers, heads=heads)
        self.head = nn.Sequential(nn.Linear(width + num_actions, width), nn.GELU(), nn.Linear(width, 1))

    def forward_query(self, tokens, reset, valid, action_idx):
        feat = self.core.sequence(tokens, reset=reset, valid=valid)[-1]
        onehot = F.one_hot(action_idx, num_classes=self.num_actions).to(feat.dtype)
        return self.head(torch.cat((feat, onehot), dim=-1)).squeeze(-1)


class HistoryVCritic(nn.Module):
    def __init__(self, input_dim, width=128, layers=2, heads=4, memory_length=4):
        super().__init__()
        self.core = CausalDecoderCore(input_dim, width=width, memory_length=memory_length, layers=layers, heads=heads)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward_query(self, tokens, reset, valid):
        feat = self.core.sequence(tokens, reset=reset, valid=valid)[-1]
        return self.head(feat).squeeze(-1)


def zero_redundant_key_bias_grad(core):
    for block in core.layers:
        bias = block.key_value.bias
        if bias.grad is not None:
            bias.grad[:bias.numel() // 2].zero_()


def importance_weights(advantage, temperature, max_weight=100.0):
    return torch.exp((advantage / temperature).clamp(min=-20.0, max=math.log(max_weight))).detach()


def binary_auc(scores, labels):
    pos = [float(s) for s, y in zip(scores, labels) if y > 0.5]
    neg = [float(s) for s, y in zip(scores, labels) if y <= 0.5]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def make_sequence_tensors(trials, norm_stats, *, suite, variant='own', device='cpu'):
    """Build causal [T=2, N, D] token tensors, reset masks, and valid masks."""
    mean_s, std_s = norm_stats['state_mean'], norm_stats['state_std']
    n = len(trials)
    if suite in ('probed', 'probed_action_reward'):
        suite = 'probed'
        dim = 20  # 16 state + u + r_signed + done + start
    elif suite in ('drift', 'uncompensated_drift'):
        suite = 'drift'
        dim = 22  # 16 state + u + delta_y + cmd_y + r_signed + done + start
        dm, ds = norm_stats['drift_mean'], norm_stats['drift_std']
    else:
        raise ValueError(suite)
    tokens = np.zeros((2, n, dim), dtype=np.float32)
    reset = np.zeros((2, n), dtype=np.bool_)
    valid = np.ones((2, n), dtype=np.bool_)
    reset[0, :] = True
    for i, t in enumerate(trials):
        if variant == 'own':
            s_state, u1, r1 = t['support_state'], t['u_sup'], 2.0 * t['r_sup'] - 1.0
            dy, cy = t.get('delta_y', 0.0), t.get('cmd_y', 0.0)
        elif variant == 'matched_donor':
            s_state, u1, r1 = t['donor_state'], t['donor_u_sup'], 2.0 * t['donor_r_sup'] - 1.0
            dy, cy = t.get('donor_delta_y', 0.0), t.get('donor_cmd_y', 0.0)
        elif variant == 'cleared':
            s_state, u1, r1 = t['support_state'], t['u_sup'], 2.0 * t['r_sup'] - 1.0
            dy, cy = t.get('delta_y', 0.0), t.get('cmd_y', 0.0)
            reset[1, i] = True
        elif variant == 'feedback_masked':
            s_state, u1, r1 = t['support_state'], t['u_sup'], 0.0
            dy, cy = t.get('delta_y', 0.0), t.get('cmd_y', 0.0)
        elif variant == 'counterfactual_reward':
            s_state, u1, r1 = t['support_state'], t['u_sup'], 1.0 - 2.0 * t['r_sup']
            dy, cy = t.get('delta_y', 0.0), t.get('cmd_y', 0.0)
        else:
            raise ValueError(variant)
        qs_norm = np.clip((t['query_state'] - mean_s) / std_s, -12.0, 12.0)
        ss_norm = np.clip((s_state - mean_s) / std_s, -12.0, 12.0)
        if suite == 'probed':
            if variant != 'cleared':
                tokens[0, i, :16] = ss_norm
                tokens[0, i, 16] = float(u1)
                tokens[0, i, 17] = float(r1)
                tokens[0, i, 18] = 1.0  # done
                tokens[0, i, 19] = 1.0  # start
            else:
                # Match episode_reset_batch: zero support token fields as well as setting reset[1]=True.
                tokens[0, i, 19] = 1.0
            tokens[1, i, :16] = qs_norm
            tokens[1, i, 18] = 0.0  # done
            tokens[1, i, 19] = 1.0  # start
        else:
            if variant != 'cleared':
                drift_n = np.clip((np.array([dy, cy], dtype=np.float32) - dm) / ds, -12.0, 12.0)
                tokens[0, i, :16] = ss_norm
                tokens[0, i, 16] = float(u1)
                tokens[0, i, 17:19] = drift_n
                tokens[0, i, 19] = float(r1)
                tokens[0, i, 20] = 1.0
                tokens[0, i, 21] = 1.0
            else:
                tokens[0, i, 21] = 1.0
            tokens[1, i, :16] = qs_norm
            tokens[1, i, 20] = 0.0
            tokens[1, i, 21] = 1.0
    return (
        torch.from_numpy(tokens).to(device),
        torch.from_numpy(reset).to(device),
        torch.from_numpy(valid).to(device),
    )


def verify_incremental_parity(core, tokens, reset, valid):
    """Verify CausalDecoderCore.forward() incremental cache matches .sequence()."""
    core.eval()
    with torch.no_grad():
        seq_out = core.sequence(tokens, reset=reset, valid=valid)
        state = core.initial_state(tokens.shape[1])
        inc_out = []
        for t in range(tokens.shape[0]):
            empty = core.initial_state(tokens.shape[1])
            r = reset[t]
            state = {
                k: torch.where(r.reshape(-1, *([1] * (v.ndim - 1))), empty[k], v)
                for k, v in state.items()
            }
            feat, state = core(tokens[t], state)
            inc_out.append(feat)
        inc_out = torch.stack(inc_out, dim=0)
        max_diff = float((seq_out - inc_out).abs().max().item())
        assert max_diff <= 1e-5, f'Incremental vs sequence decoder mismatch: {max_diff}'
    return max_diff


def run_suite(suite_name, train_trials_base, eval_cross, eval_same, train_scenes, eval_scenes, *,
              norm_stats, out_dir, device):
    """Fit 3 seeds of critics + 6 actors and evaluate across checkpoints and interventions."""
    # Expand each two-episode trial with uniform behavior actions u_2 in (-1, 0, +1).
    train_behavior = []
    for base in train_trials_base:
        for a_idx, u2 in enumerate(ACTIONS):
            train_behavior.append(dict(base, u2=u2, a2_idx=a_idx, r2=float(base['outcomes'][u2])))
    eval_behavior = []
    for base in eval_cross:
        for a_idx, u2 in enumerate(ACTIONS):
            eval_behavior.append(dict(base, u2=u2, a2_idx=a_idx, r2=float(base['outcomes'][u2])))

    tr_tok, tr_rst, tr_val = make_sequence_tensors(train_behavior, norm_stats, suite=suite_name, variant='own', device=device)
    tr_a2 = torch.tensor([x['a2_idx'] for x in train_behavior], dtype=torch.long, device=device)
    tr_r2 = torch.tensor([x['r2'] for x in train_behavior], dtype=torch.float32, device=device)
    tr_u_star = torch.tensor([ACTIONS.index(-x['z']) for x in train_behavior], dtype=torch.long, device=device)

    ev_variants = {
        v: make_sequence_tensors(eval_behavior, norm_stats, suite=suite_name, variant=v, device=device)
        for v in ('own', 'matched_donor', 'cleared', 'feedback_masked')
    }
    ev_a2 = torch.tensor([x['a2_idx'] for x in eval_behavior], dtype=torch.long, device=device)
    ev_r2 = torch.tensor([x['r2'] for x in eval_behavior], dtype=torch.float32, device=device)
    # Donor action substitution: shift action index cyclically (+1 mod 3) so action differs from real u2.
    ev_donor_a2 = (ev_a2 + 1) % 3

    panel_variants = {}
    for panel_name, trials in (('eval_cross', eval_cross), ('eval_same', eval_same)):
        panel_variants[panel_name] = {
            v: make_sequence_tensors(trials, norm_stats, suite=suite_name, variant=v, device=device)
            for v in ('own', 'matched_donor', 'cleared', 'feedback_masked', 'counterfactual_reward')
        }

    input_dim = tr_tok.shape[-1]
    suite_result = dict(
        suite=suite_name,
        input_dim=input_dim,
        unique_training_queries=len(train_behavior),
        post_success_queries=0,
        seeds={},
    )

    for seed in SEEDS:
        seed_dir = out_dir / suite_name / f'seed-{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(1000 + seed)
        np.random.seed(1000 + seed)

        # 1. Fit independent causal Q and V critics on training behavior trials.
        q_net = HistoryQCritic(input_dim).to(device)
        v_net = HistoryVCritic(input_dim).to(device)
        val_opt = torch.optim.AdamW(list(q_net.parameters()) + list(v_net.parameters()), lr=3e-4, weight_decay=1e-2)
        val_rng = np.random.default_rng(5000 + seed)
        n_train = len(train_behavior)
        for _ in range(500):
            idx = torch.from_numpy(val_rng.integers(n_train, size=32)).to(device)
            val_opt.zero_grad(set_to_none=True)
            q_pred = q_net.forward_query(tr_tok[:, idx], tr_rst[:, idx], tr_val[:, idx], tr_a2[idx])
            v_pred = v_net.forward_query(tr_tok[:, idx], tr_rst[:, idx], tr_val[:, idx])
            loss = F.mse_loss(q_pred, tr_r2[idx]) + F.mse_loss(v_pred, tr_r2[idx])
            loss.backward()
            zero_redundant_key_bias_grad(q_net.core)
            zero_redundant_key_bias_grad(v_net.core)
            nn.utils.clip_grad_norm_(list(q_net.parameters()) + list(v_net.parameters()), 1.0, error_if_nonfinite=True)
            val_opt.step()

        q_net.eval().requires_grad_(False)
        v_net.eval().requires_grad_(False)
        parity_q = verify_incremental_parity(q_net.core, tr_tok[:, :8], tr_rst[:, :8], tr_val[:, :8])
        parity_v = verify_incremental_parity(v_net.core, tr_tok[:, :8], tr_rst[:, :8], tr_val[:, :8])

        with torch.no_grad():
            tr_q = q_net.forward_query(tr_tok, tr_rst, tr_val, tr_a2)
            tr_v = v_net.forward_query(tr_tok, tr_rst, tr_val)
            tr_adv = tr_q - tr_v
            r_mean = float(tr_r2.mean().item())
            weights_map = dict(
                uniform_bc=torch.ones_like(tr_adv),
                dit_tau1=importance_weights(tr_adv, 1.0),
                dit_tau01=importance_weights(tr_adv, 0.1),
                known_outcome_tau01=importance_weights(tr_r2 - r_mean, 0.1),
            )
            ev_q = q_net.forward_query(*ev_variants['own'], ev_a2)
            ev_v = v_net.forward_query(*ev_variants['own'])
            ev_adv = ev_q - ev_v
            ev_q_donor_act = q_net.forward_query(*ev_variants['own'], ev_donor_a2)
            ev_q_donor_hist = q_net.forward_query(*ev_variants['matched_donor'], ev_a2)
            ev_q_cleared = q_net.forward_query(*ev_variants['cleared'], ev_a2)
            ev_q_masked = q_net.forward_query(*ev_variants['feedback_masked'], ev_a2)

        def weight_stats(w_tensor, r_tensor):
            w = w_tensor.detach().cpu().numpy().astype(np.float64)
            r = r_tensor.detach().cpu().numpy().astype(np.float64)
            ess = float(w.sum() ** 2 / (len(w) * (w * w).sum()))
            shift_pp = float(100.0 * ((w * r).sum() / w.sum() - r.mean()))
            return dict(ess_fraction=ess, success_share_shift_pp=shift_pp)

        ev_w_tau1 = importance_weights(ev_adv, 1.0)
        ev_w_tau01 = importance_weights(ev_adv, 0.1)
        ev_w_known = importance_weights(ev_r2 - r_mean, 0.1)
        ev_mse_real = float(((ev_q - ev_r2) ** 2).mean().item())
        critic_diag = dict(
            incremental_parity_max_diff=max(parity_q, parity_v),
            train_auc=binary_auc(tr_adv.cpu().numpy(), tr_r2.cpu().numpy()),
            eval_auc=binary_auc(ev_adv.cpu().numpy(), ev_r2.cpu().numpy()),
            eval_q_mse=ev_mse_real,
            eval_delta_mse_donor_action=float(((ev_q_donor_act - ev_r2) ** 2).mean().item() - ev_mse_real),
            eval_delta_mse_donor_history=float(((ev_q_donor_hist - ev_r2) ** 2).mean().item() - ev_mse_real),
            eval_delta_mse_cleared=float(((ev_q_cleared - ev_r2) ** 2).mean().item() - ev_mse_real),
            eval_delta_mse_feedback_masked=float(((ev_q_masked - ev_r2) ** 2).mean().item() - ev_mse_real),
            train_weights={k: weight_stats(v, tr_r2) for k, v in weights_map.items()},
            eval_weights=dict(
                dit_tau1=weight_stats(ev_w_tau1, ev_r2),
                dit_tau01=weight_stats(ev_w_tau01, ev_r2),
                known_outcome_tau01=weight_stats(ev_w_known, ev_r2),
            ),
        )

        # 2. Fit all 6 actor models from identical initialization per seed and identical sampler schedule.
        init_actor = HistoryActor(input_dim).to(device)
        init_state = {k: v.clone() for k, v in init_actor.state_dict().items()}
        seed_methods = {}
        for m_idx, method in enumerate(METHODS):
            actor = HistoryActor(input_dim).to(device)
            actor.load_state_dict(init_state)
            opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-2)
            sampler = np.random.default_rng(10000 * seed + 100 + m_idx)
            seen = np.zeros(n_train, dtype=np.int32)
            is_teacher = method in ('dpt', 'memoryless')
            targets_t = tr_u_star if is_teacher else tr_a2
            weights_t = torch.ones_like(tr_adv) if is_teacher else weights_map[method]
            tokens_in = tr_tok.clone()
            reset_in = tr_rst.clone()
            if method == 'memoryless':
                tokens_in[0, :, :-1] = 0.0
                reset_in[1, :] = True

            checkpoints = {}
            for step in range(1, MILESTONES[-1] + 1):
                actor.train()
                batch_idx = sampler.integers(n_train, size=32)
                seen[batch_idx] += 1
                idx_t = torch.from_numpy(batch_idx).to(device)
                opt.zero_grad(set_to_none=True)
                logits = actor.sequence_logits(tokens_in[:, idx_t], reset_in[:, idx_t], tr_val[:, idx_t])[-1]
                ce = F.cross_entropy(logits, targets_t[idx_t], reduction='none')
                loss = (ce * weights_t[idx_t]).mean()
                loss.backward()
                zero_redundant_key_bias_grad(actor.core)
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()

                if step in MILESTONES:
                    actor.eval()
                    parity_actor = verify_incremental_parity(actor.core, tokens_in[:, :8], reset_in[:, :8], tr_val[:, :8])
                    with torch.no_grad():
                        eval_out = {}
                        for panel_name, trials in (('eval_cross', eval_cross), ('eval_same', eval_same)):
                            panel_res = {}
                            preds_by_var = {}
                            for var, (vt, vr, vv) in panel_variants[panel_name].items():
                                if method == 'memoryless':
                                    vt, vr, vv = panel_variants[panel_name]['cleared']
                                pred_idx = actor.sequence_logits(vt, vr, vv)[-1].argmax(-1).cpu().tolist()
                                preds_by_var[var] = [ACTIONS[k] for k in pred_idx]
                                panel_res[var] = summarize_trials(trials, eval_scenes, preds_by_var[var])
                            for cmp_name, other_var in (
                                ('own_vs_donor', 'matched_donor'),
                                ('own_vs_cleared', 'cleared'),
                                ('own_vs_masked', 'feedback_masked'),
                                ('own_vs_cf_reward', 'counterfactual_reward'),
                            ):
                                panel_res[cmp_name] = summarize_trials(
                                    trials, eval_scenes, preds_by_var['own'],
                                    baseline_predictions=preds_by_var[other_var],
                                )
                            eval_out[panel_name] = panel_res
                    checkpoints[str(step)] = dict(
                        step=step,
                        incremental_parity_max_diff=parity_actor,
                        unique_queries_covered=int((seen > 0).sum()),
                        coverage_fraction=float((seen > 0).mean()),
                        eval=eval_out,
                    )
                    if step == MILESTONES[-1]:
                        ckpt_path = seed_dir / f'{method}-step-{step:04d}.pt'
                        torch.save(dict(
                            step=step, seed=seed, method=method, suite=suite_name,
                            model=actor.state_dict(), optimizer=opt.state_dict(),
                            critic_q=q_net.state_dict(), critic_v=v_net.state_dict(),
                            rng=dict(sampler=sampler.bit_generator.state, torch=torch.get_rng_state()),
                            norm_stats={k: v.tolist() for k, v in norm_stats.items()},
                            seen_counts=seen.tolist(),
                        ), ckpt_path)
            seed_methods[method] = dict(
                final_coverage_fraction=float((seen > 0).mean()),
                unique_queries_covered=int((seen > 0).sum()),
                total_draws=int(seen.sum()),
                checkpoints=checkpoints,
            )
        suite_result['seeds'][str(seed)] = dict(critic=critic_diag, methods=seed_methods)
    return suite_result


def aggregate_stage2(suite_result):
    """Aggregate 3-seed mean and per-seed ranges across checkpoints and conditions."""
    agg = {}
    for step in MILESTONES:
        skey = str(step)
        agg[skey] = {}
        for method in METHODS:
            m_res = {}
            for panel in ('eval_cross', 'eval_same'):
                p_res = {}
                for metric_key in ('own', 'matched_donor', 'cleared', 'feedback_masked', 'counterfactual_reward',
                                   'own_vs_donor', 'own_vs_cleared', 'own_vs_masked'):
                    att2_vals = [
                        suite_result['seeds'][str(s)]['methods'][method]['checkpoints'][skey]['eval'][panel][metric_key]['attempt2_success']['point']
                        for s in SEEDS
                    ]
                    cum_vals = [
                        suite_result['seeds'][str(s)]['methods'][method]['checkpoints'][skey]['eval'][panel][metric_key]['cumulative_success']['point']
                        for s in SEEDS
                    ]
                    row = dict(
                        attempt2_mean=float(np.mean(att2_vals)),
                        attempt2_per_seed=att2_vals,
                        cumulative_mean=float(np.mean(cum_vals)),
                        cumulative_per_seed=cum_vals,
                    )
                    if metric_key.startswith('own_vs_'):
                        diff_vals = [
                            suite_result['seeds'][str(s)]['methods'][method]['checkpoints'][skey]['eval'][panel][metric_key]['diff_attempt2']['point']
                            for s in SEEDS
                        ]
                        diff_cum_vals = [
                            suite_result['seeds'][str(s)]['methods'][method]['checkpoints'][skey]['eval'][panel][metric_key]['diff_cumulative']['point']
                            for s in SEEDS
                        ]
                        row['diff_attempt2_mean'] = float(np.mean(diff_vals))
                        row['diff_attempt2_per_seed'] = diff_vals
                        row['diff_cumulative_mean'] = float(np.mean(diff_cum_vals))
                        row['diff_cumulative_per_seed'] = diff_cum_vals
                    p_res[metric_key] = row
                m_res[panel] = p_res
            agg[skey][method] = m_res
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    assert sys.platform == 'linux'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    summary = json.loads((args.panel / 'SUMMARY.json').read_text())
    admitted, by_key, repeat_checks = build_panel_records(summary)
    train_scenes, eval_scenes = admitted[:8], admitted[8:16]
    assert set(train_scenes).isdisjoint(eval_scenes)

    stage1 = evaluate_stage1(train_scenes, eval_scenes, by_key)

    # Fit state & drift normalization ONLY on training scenes' nominal / uncompensated states.
    train_states = np.stack([np.asarray(by_key[(s, 'nominal')]['decision_state'], dtype=np.float32) for s in train_scenes])
    train_drift_vals = np.asarray([
        [by_key[(s, arm_name(z, 0))]['public_delta_eef_y'], by_key[(s, arm_name(z, 0))]['commanded_mean_y']]
        for s in train_scenes for z in (-1, 1)
    ], dtype=np.float32)
    norm_stats = dict(
        state_mean=train_states.mean(axis=0),
        state_std=train_states.std(axis=0).clip(0.05),
        drift_mean=train_drift_vals.mean(axis=0),
        drift_std=train_drift_vals.std(axis=0).clip(1e-3),
    )

    # Probed action+reward suite.
    probed_train_base = build_probed_trials(train_scenes, by_key, mode='all_ordered')
    probed_eval_cross = build_probed_trials(eval_scenes, by_key, mode='cross_scene')
    probed_eval_same = build_probed_trials(eval_scenes, by_key, mode='same_scene')
    probed_res = run_suite(
        'probed_action_reward', probed_train_base, probed_eval_cross, probed_eval_same,
        train_scenes, eval_scenes, norm_stats=norm_stats, out_dir=args.out, device=device,
    )
    probed_res['aggregate'] = aggregate_stage2(probed_res)

    # Uncompensated drift suite (u_sup = 0, support reward = 0 everywhere when two-way holds).
    drift_train_base = build_drift_trials(train_scenes, by_key, mode='all_ordered')
    drift_eval_cross = build_drift_trials(eval_scenes, by_key, mode='cross_scene')
    drift_eval_same = build_drift_trials(eval_scenes, by_key, mode='same_scene')
    drift_res = run_suite(
        'uncompensated_drift', drift_train_base, drift_eval_cross, drift_eval_same,
        train_scenes, eval_scenes, norm_stats=norm_stats, out_dir=args.out, device=device,
    )
    drift_res['aggregate'] = aggregate_stage2(drift_res)

    result = dict(
        completed=True,
        device=device,
        panel_summary_sha256=sha256_file(args.panel / 'SUMMARY.json'),
        train_scenes=train_scenes,
        eval_scenes=eval_scenes,
        repeat_checks=repeat_checks,
        stage1=stage1,
        stage2=dict(probed_action_reward=probed_res, uncompensated_drift=drift_res),
        elapsed_seconds=time.time() - started,
    )
    write_json(args.out / 'RESULT.json', result)
    print(json.dumps(dict(
        completed=True,
        device=device,
        elapsed_seconds=result['elapsed_seconds'],
        stage1_eval_cross_own=stage1['probed_eval_cross']['feedback_own']['attempt2_success']['point'],
        stage1_eval_cross_donor=stage1['probed_eval_cross']['feedback_matched_donor']['attempt2_success']['point'],
        stage1_eval_cross_obs=stage1['probed_eval_cross']['current_obs_ref']['attempt2_success']['point'],
        stage1_eval_cross_fixed=stage1['probed_eval_cross']['best_fixed']['attempt2_success']['point'],
        stage1_drift_own=stage1['drift_eval_cross']['drift_own']['attempt2_success']['point'],
        stage1_drift_donor=stage1['drift_eval_cross']['drift_matched_donor']['attempt2_success']['point'],
    ), indent=2), flush=True)


if __name__ == '__main__':
    main()
