"""Allocated Linux CPU only; synthetic scene outcomes are not robot evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.steering_condition_cross import (
    TRAIN1, CONDITIONS, ACTOR_SEED, _FrozenEpisodeController, _frozen_identity,
    run_condition_cross, summarize_cross,
)
from fluffy_bassoon.meta_rl.steering_condition_collection import _state_digest
from fluffy_bassoon.meta_rl.robotwin_history_corpus import parameter_digest
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC, SACConfig
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter
from fluffy_bassoon.rl.robotwin_noise import _observation_hash
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution
from test_steering_actuator_condition import make_pair
from test_steering_condition_collection import profile


BLIND = "one_val_selected_context_blind"


def artifact(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def policy(seed):
    return NoiseSAC(2062, 4, action_limit=1., device="cpu", seed=seed,
        config=SACConfig(hidden_dims=(8,), layer_norm=False, replay_capacity=64,
                         batch_size=32, warmup_chunks=32, utd=4))


def summary_rows():
    rows = []
    for condition in CONDITIONS:
        matched = TRAIN1[CONDITIONS.index(condition)]
        for case in range(4):
            for method in (*TRAIN1, BLIND):
                # Minus matched and fixed blind tie; plus matched wins once.
                success = case == 0 and (method == matched or (method == BLIND and condition == CONDITIONS[0]))
                rows.append(dict(spec=dict(id=f"{condition}:{case}:{method}", condition_id=condition,
                    case=case, method=method), complete=True, success=success, primitive_steps=3))
    return rows


def test_declared_aggregate_blind_threshold_allows_one_condition_to_tie_blind():
    result = summarize_cross(summary_rows(), BLIND)
    assert result["criterion_pass"] and result["both_directions_pass"]
    assert result["aggregate_blind"]["success_gap"] == .125
    assert result["directions"][CONDITIONS[0]]["paired"]["blind"]["success_gap"] == 0
    assert all(row["paired"]["mismatched"]["success_gap"] == .25 for row in result["directions"].values())


@pytest.mark.parametrize("fault", ["missing", "duplicate", "partial", "wrong_condition"])
def test_summary_requires_all24_complete_predeclared_cells(fault):
    rows = summary_rows()
    if fault == "missing": rows.pop()
    if fault == "duplicate": rows[-1] = deepcopy(rows[0])
    if fault == "partial": rows[-1]["complete"] = False
    if fault == "wrong_condition": rows[-1]["spec"]["condition_id"] = "foreign"
    with pytest.raises(ValueError): summarize_cross(rows, BLIND)


def test_one_failed_matched_mismatch_direction_cannot_pass_on_aggregate_blind_only():
    rows = summary_rows()
    for row in rows:
        if row["spec"]["condition_id"] == CONDITIONS[1] and row["spec"]["method"] == TRAIN1[1]:
            row["success"] = False
    result = summarize_cross(rows, BLIND)
    assert not result["criterion_pass"] and not result["both_directions_pass"]


def test_sampled_actor_draws_pair_privately_without_source_rng_replay_or_optimizer_changes():
    actor = policy(77)
    before, global_rng = _frozen_identity(actor), torch.get_rng_state().clone()
    ref = dict(policy_sha256=parameter_digest(actor.actor.state_dict()), optimizer_updates=actor.updates,
               sampling_mode="sampled_gaussian")
    left = _FrozenEpisodeController(actor, TRAIN1[0], ref, CONDITIONS[0], lambda: None, ACTOR_SEED)
    right = _FrozenEpisodeController(actor, TRAIN1[0], ref, CONDITIONS[1], lambda: None, ACTOR_SEED)
    obs = np.zeros(2062, np.float32)
    np.testing.assert_array_equal(left.act(obs), right.act(obs))
    assert torch.equal(left.last_rng["before"], right.last_rng["before"])
    assert torch.equal(left.last_rng["after"], right.last_rng["after"])
    first = deepcopy(left.last_rng)
    left.act(obs)
    assert not torch.equal(first["before"], left.last_rng["before"])
    left.observe(obs, np.zeros(4), 1., obs, True, False, 3, None)
    assert _frozen_identity(actor) == before and torch.equal(torch.get_rng_state(), global_rng)
    assert left.decision_metadata()["cross_actor_sampling_mode"] == "sampled_gaussian"


class SyntheticControl:
    """Test only: existing corpus/runner own all records and action intents."""
    native_horizon = 3
    def __init__(self, root, p, distribution):
        self.env = SimpleNamespace(corpus_writer=RobotwinCorpusWriter(root, {}))
        self.post_command_transform = distribution
        self.condition_contract = distribution.contract()
        self.interface_sha256, self.p = p.interface_sha256, p
        self.active, self.elapsed_steps, self.calls = False, 0, []
        self.coefficients = np.zeros(4, np.float32)
    def episode_limit_for_task(self, task): return self.p.episode_limit
    def mapping_sha256(self): return self.p.mapping_sha256
    def set_coefficients(self, arm, values): self.coefficients = np.asarray(values, np.float32).copy()
    def set_recording_context(self, values):
        self.env.corpus_writer.set_context(values)
    def reset(self, task, seed, *, motor_seed, paired_instruction=None):
        assert not self.active
        self.calls.append(dict(seed=seed, motor_seed=motor_seed, instruction=paired_instruction))
        self.features = np.full(2062, (seed % 100000)/100000, np.float32)
        self.raw = dict(main_images=np.arange(12, dtype=np.uint8).reshape(1,2,2,3),
            wrist_images=np.arange(24, dtype=np.uint8).reshape(1,2,2,2,3),
            states=np.full((1,14), float(seed)),
            task_descriptions=list(paired_instruction or [f"actual instruction:{seed}"]))
        info = dict(task=task, requested_seed=seed, actual_seed=seed, instruction=self.raw["task_descriptions"],
            observation_sha256=_observation_hash(self.raw),
            features_sha256=hashlib.sha256(self.features.tobytes()).hexdigest(), reset_policy_actions=0)
        info["corpus_episode_id"] = self.env.corpus_writer.begin_episode(self.raw, self.features, info)
        self.last_reset_info = info
        self.active, self.elapsed_steps = True, 0
        return self.features.copy()
    def record_decision_metadata(self, values): self.metadata = values
    def step(self, coefficients):
        assert self.active
        self.coefficients = np.asarray(coefficients, np.float32).copy()
        writer = self.env.corpus_writer
        writer.begin_decision(np.zeros((3,32), np.float32), np.zeros((3,14), np.float32), self.metadata)
        condition = 0 if self.post_command_transform.selected_sign < 0 else 1
        method = writer.context["method"]
        success = method == TRAIN1[condition] or (method == BLIND and condition == 0)
        writer.complete_native(next_raw=self.raw, next_features=self.features, raw_rewards=[float(success)],
            raw_terminations=[success], raw_truncations=[not success], valid_action_mask=[1,1,1],
            reward=float(success), terminated=success, truncated=not success,
            executed_steps=3, success=success, duration_seconds=.01)
        chunk = writer.finalize_learning_flags(reward=float(success), terminated=success, truncated=not success,
            budget_truncated=False, episode_primitive_steps=3)
        self.active, self.elapsed_steps = False, 3
        return self.features.copy(), float(success), success, not success, dict(
            primitive_steps=3, success=success, corpus_chunk=chunk)
    def snapshot(self): return dict(elapsed_steps=self.elapsed_steps)


@pytest.fixture
def setup_cross(tmp_path):
    p = replace(profile(), episode_limit=3)
    distribution = CalibrationDistribution(make_pair()[1])
    policies = {TRAIN1[0]:policy(10), TRAIN1[1]:policy(11)}
    policies[BLIND] = policies[TRAIN1[0]]
    references = {}
    for method in TRAIN1:
        value = policies[method]
        # The production owner supplies immutable full source publications.
        references[method] = dict(artifact(tmp_path/(method+'.json'), dict(synthetic_fixture=True)),
            policy_sha256=parameter_digest(value.actor.state_dict()), optimizer_updates=value.updates,
            sampling_mode="sampled_gaussian", source_method=method)
    references[BLIND] = deepcopy(references[TRAIN1[0]])
    val = artifact(tmp_path/'source-validation.json', dict(synthetic_fixture=True))
    blind = artifact(tmp_path/'blind.json', dict(selected_method=BLIND, selected_source_method=TRAIN1[0],
        selected_from="source_validation", cross_cases_opened=False, controller_ref=references[BLIND],
        final_validation_successes=dict.fromkeys(TRAIN1,1), source_validation_refs=[val]))
    control = SyntheticControl(tmp_path/'corpus', p, distribution)
    calls = []
    def checkpoint(event, payload, *, immutable):
        calls.append((event, deepcopy(payload)))
        return dict(path='owner-recovery', sha256=_state_digest(payload))
    kwargs = dict(run_id="test", output=tmp_path/'cross', guard=lambda: None,
        controller_refs=references, context_blind_method=BLIND, blind_selection_ref=blind,
        source_reset_identities=[dict(actual_seed=1, observation_sha256='e'*64, features_sha256='f'*64)],
        checkpoint=checkpoint)
    ledger = dict(root=tmp_path/'ledger', campaign_id='test', total_actions=300000, source_caps=p.source_caps(),
        phase_caps=p.contract()['phase_caps'], contract=dict(scope_sha256=p.scope_sha256),
        run_id='test', source_commit='a'*40)
    return control, p, distribution, policies, kwargs, ledger, calls


def test_actual_common_runner_and_corpus_complete24_cells_pair_prompts_and_freeze_all_source_state(setup_cross):
    control,p,distribution,policies,kwargs,ledger,calls = setup_cross
    before = {k:_frozen_identity(v) for k,v in policies.items()}
    with CampaignBudget(**ledger) as budget:
        result = run_condition_cross(control,budget,p,distribution,policies,**kwargs)
        assert budget.snapshot()['actual_actions'] == 72
        assert budget.snapshot()['by_phase'] == dict(condition_cross=72)
    assert result['complete'] and result['criterion_pass'] and len(result['episode_ids']) == 24
    assert all(_frozen_identity(policies[k]) == before[k] for k in policies)
    for offset in range(0,24,6):
        rows = control.calls[offset:offset+6]
        assert rows[0]['instruction'] is None
        assert len({r['seed'] for r in rows}) == len({r['motor_seed'] for r in rows}) == 1
        assert all(r['instruction'] == (f"actual instruction:{rows[0]['seed']}",) for r in rows[1:])
    assert calls[-1][0] == 'condition_cross_complete'


@pytest.mark.parametrize('fault',['blind_after_open','blind_wrong_source','sampling','source_overlap','wrong_mapping','tampered_ref'])
def test_cross_rejects_unbound_or_leaking_setup_before_any_command(setup_cross, fault):
    control,p,distribution,policies,kwargs,ledger,_ = setup_cross
    if fault.startswith('blind_'):
        path=Path(kwargs['blind_selection_ref']['path']); value=json.loads(path.read_text())
        if fault=='blind_after_open': value['cross_cases_opened']=True
        else: value['selected_source_method']=TRAIN1[1]
        kwargs['blind_selection_ref']=artifact(path,value)
    if fault=='sampling': kwargs['controller_refs'][TRAIN1[0]]['sampling_mode']='deterministic_mean'
    if fault=='wrong_mapping': control.mapping_sha256=lambda:'f'*64
    if fault=='tampered_ref': Path(kwargs['controller_refs'][TRAIN1[0]]['path']).write_text('changed')
    if fault=='source_overlap': kwargs['source_reset_identities'][0]['actual_seed']=2300000000
    with CampaignBudget(**ledger) as budget:
        with pytest.raises(ValueError): run_condition_cross(control,budget,p,distribution,policies,**kwargs)
        assert budget.snapshot()['actual_actions']==0 and not budget.snapshot()['unresolved_decisions']


def test_cross_does_not_resume_failed_or_pending_episode_state(setup_cross):
    control,p,distribution,policies,kwargs,ledger,_=setup_cross
    kwargs['state']=dict(format='source_only_frozen_condition_cross_v1',plan_sha256='a'*64,
                         status='blocked_failure',pending={})
    with CampaignBudget(**ledger) as budget:
        with pytest.raises(ValueError,match='closed boundary'):
            run_condition_cross(control,budget,p,distribution,policies,**kwargs)
    assert not control.calls


def test_owner_pre_rpc_boundary_stops_without_pending_then_resumes_exact_next_cell(setup_cross):
    control,p,distribution,policies,kwargs,ledger,calls=setup_cross
    admissions=iter([True,True,False])
    kwargs['should_start_episode']=lambda:next(admissions)
    with CampaignBudget(**ledger) as budget:
        stopped=run_condition_cross(control,budget,p,distribution,policies,**kwargs)
        assert stopped['status']=='boundary_stop' and stopped['complete'] is False
        assert len(control.calls)==len(stopped['episode_ids'])==2
        assert budget.snapshot()['actual_actions']==6 and not budget.snapshot()['unresolved_decisions']
        event,saved=calls[-1]
        assert event=='condition_cross_boundary_stop' and saved['pending'] is None and not control.active
        assert 'failure' not in saved and saved['status']=='boundary_stop'
        kwargs['state']=saved
        kwargs['should_start_episode']=lambda:True
        result=run_condition_cross(control,budget,p,distribution,policies,**kwargs)
        assert result['complete'] and len(control.calls)==24
        assert budget.snapshot()['actual_actions']==72
        committed=[payload['records'][-1]['spec']['id'] for event,payload in calls
                   if event=='condition_cross_episode_committed']
        assert len(committed)==len(set(committed))==24
