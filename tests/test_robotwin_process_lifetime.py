"""Real driver lifecycle control flow with CPU-only simulator/learner fakes.

Run on allocated Linux after integrating the production owner/recovery driver.
The nested functions are copied from its AST; only Python closure bindings
become a test namespace, so no allocated-runtime gate or model is constructed.
"""
from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest


DRIVER = Path(__file__).parents[1] / "scripts/diagnostics/robotwin_noise_compare.py"
SOURCE_TASKS = ("handover_mic", "pick_diverse_bottles", "place_a2b_left",
                "place_object_stand", "place_phone_stand", "scan_object")
TARGET_TASKS = ("adjust_bottle", "move_stapler_pad")


class WallBudgetReached(RuntimeError):
    pass


class _ClosureBindings(ast.NodeTransformer):
    def visit_Nonlocal(self, node):
        return ast.copy_location(ast.Global(names=node.names), node)


def bind_driver_functions(namespace):
    source = ast.parse(DRIVER.read_text())
    main = next(node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    names = {"guard", "native", "evaluate_frozen_sources", "evaluate_stage"}
    functions = [_ClosureBindings().visit(deepcopy(node)) for node in main.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(DRIVER), "exec"), namespace)
    return namespace


@pytest.fixture
def lifecycle(tmp_path):
    def make(method="ppo"):
        rng = random.Random(37)
        events = []

        class Environment:
            resets = 0

            def reset(self, task, seed):
                assert namespace["vector_env"]._pool is None
                self.resets += 1
                events.append(("reset", task, seed))
                rng.random()  # A simulator may consume process RNG.
                return np.zeros(2, np.float32)

            def step(self, noise):
                return np.zeros(2, np.float32), 0., True, False, dict(
                    primitive_steps=1, success=False)

        class Learner:
            updates = 50

            def state_dict(self):
                return dict(trained=True, updates=self.updates)

            def load_state_dict(self, state):
                assert state == self.state_dict()

            def collect_trial(self, env, task, seeds, **kwargs):
                rows = []
                for seed in seeds:
                    env.reset(task, seed)
                    rows.append(dict(success=False))
                return None, dict(episodes=rows, primitive_steps=len(rows))

        args = SimpleNamespace(output=tmp_path / method, wall_seconds=3600,
            boundary_reserve_seconds=300, max_source_trials_per_process=6,
            recycle_at_round_boundary=True, ppo_trials_per_update=6,
            seed=20261030, source_tasks=list(SOURCE_TASKS),
            eval_tasks=list(TARGET_TASKS), source_validation_episodes=4,
            eval_episodes=8, eval_trials=1, native_episodes=1 if method == "ppo" else 0,
            methods=[method], campaign_root=tmp_path / "campaign",
            record_corpus=True, stage_source_validation_only=False)
        state = dict(steps=10_000, iterations=50, completed_trials=300,
            pending=[], selected_resume=None, task_steps={}, learner=Learner(),
            schedule=SimpleNamespace(state_dict=lambda: dict(trials=300)))
        namespace = dict(args=args, time=SimpleNamespace(monotonic=lambda: 0.),
            started=0., completed_process_trials=0, source_states={method: state},
            completed_units=set(), checkpoint_rng_override=None, stage_reports={},
            history=[], contract=dict(lineage="DB"), vector_env=SimpleNamespace(_pool=None),
            capture_rng=rng.getstate, restore_rng=rng.setstate,
            deepcopy=deepcopy, hashlib=hashlib, np=np, math=math,
            SOURCE_TASKS=SOURCE_TASKS, HELDOUT_TASKS=TARGET_TASKS,
            WallBudgetReached=WallBudgetReached, env=Environment(), noise_shape=(2,),
            observation_hash=lambda value: "feature-hash", episodes_path=tmp_path / "episodes.jsonl",
            append_json=lambda *a, **k: None, log=lambda *a, **k: None,
            set_context=lambda *a, **k: None, decision_metadata=lambda *a, **k: None,
            record_episode=lambda *a, **k: None, new_ppo=lambda *a, **k: Learner(),
            torch=SimpleNamespace(save=lambda value, path: Path(path).write_bytes(pickle.dumps(value))),
            write_json=lambda path, value: Path(path).write_text(json.dumps(value)),
            save_joint=lambda **kwargs: events.append(("checkpoint", deepcopy(namespace["checkpoint_rng_override"]))),
            events=events, test_rng=rng)
        return bind_driver_functions(namespace)
    return make


def test_full_n6_update_lifetime_stops_before_due_evaluation_and_side_effects(lifecycle):
    scope = lifecycle()
    scope["completed_process_trials"] = 6
    with pytest.raises(WallBudgetReached, match="lifetime"):
        scope["evaluate_stage"](source_iteration=50)
    assert not scope["args"].output.exists()
    assert scope["env"].resets == 0 and not scope["events"]


@pytest.mark.parametrize("iteration", [10, 25])
def test_completed_earlier_milestones_are_not_repeated_on_later_resume(lifecycle, iteration):
    scope = lifecycle()
    scope["completed_units"].add(f"stage:iteration:{iteration}:complete")
    scope["completed_process_trials"] = 6
    scope["vector_env"]._pool = object()
    scope["evaluate_stage"](source_iteration=iteration)
    assert not scope["args"].output.exists()
    assert scope["env"].resets == 0 and not scope["events"]


@pytest.mark.parametrize("method,stage_resets,with_preflight", [
    ("ppo", 76, 77), ("ppo_reset", 40, 41)])
def test_fresh_process_evaluates_original_stage_before_lazy_pool_and_restores_rng(
        lifecycle, method, stage_resets, with_preflight):
    scope = lifecycle(method)
    before = scope["capture_rng"]()
    scope["evaluate_stage"](source_iteration=50)
    assert scope["env"].resets == stage_resets
    assert scope["env"].resets + 1 == with_preflight
    assert scope["vector_env"]._pool is None
    assert scope["capture_rng"]() == before
    assert scope["checkpoint_rng_override"] is None
    assert scope["stage_reports"]["iteration:50"]["status"] == "complete"
    assert "stage:iteration:50:complete" in scope["completed_units"]
    assert scope["source_states"][method]["learner"].updates == 50
    assert scope["source_states"][method]["pending"] == []
    # A subsequent pass must return before the exhausted guard or stale pool.
    scope["completed_process_trials"] = 6
    scope["vector_env"]._pool = object()
    scope["evaluate_stage"](source_iteration=50)
    assert scope["env"].resets == stage_resets


def test_resumed_stage_skips_individually_completed_frozen_units(lifecycle):
    scope = lifecycle()
    prefix = "stage:iteration:50:"
    scope["completed_units"].update({
        prefix + "source_validation:ppo:handover_mic",
        prefix + "target:ppo:normal:adjust_bottle:0",
        prefix + "native_before_target:native:adjust_bottle:0"})
    scope["evaluate_stage"](source_iteration=50)
    assert scope["env"].resets == 76 - 4 - 8 - 1


def test_existing_pool_blocks_scalar_evaluation_before_files_or_rng_consumption(lifecycle):
    scope = lifecycle()
    scope["vector_env"]._pool = object()
    before = scope["capture_rng"]()
    with pytest.raises(WallBudgetReached, match="fresh process"):
        scope["evaluate_stage"](source_iteration=50)
    assert scope["capture_rng"]() == before
    assert not scope["args"].output.exists()
    assert scope["env"].resets == 0 and not scope["events"]


def test_failed_frozen_evaluation_restores_source_rng_without_marking_stage_complete(lifecycle):
    scope = lifecycle()
    before = scope["capture_rng"]()

    def fail(*args, **kwargs):
        scope["test_rng"].random()
        raise RuntimeError("native evaluation failed")

    scope["evaluate_frozen_sources"] = fail
    with pytest.raises(RuntimeError, match="native evaluation failed"):
        scope["evaluate_stage"](source_iteration=50)
    assert scope["capture_rng"]() == before
    assert scope["checkpoint_rng_override"] is None
    assert "stage:iteration:50:complete" not in scope["completed_units"]
    assert scope["stage_reports"]["iteration:50"]["status"] == "evaluating"
