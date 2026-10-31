import copy
from types import SimpleNamespace

import pytest

from fluffy_bassoon.meta_rl.vla_libero import sha256
from fluffy_bassoon.meta_rl.vla_rlinf_candidate_eval import (
    HISTORY_METHODS, NOISE_CONTRACT, checked_entry, history_callbacks, validate_registry,
)


def registry_and_motor():
    features = {"kind": "the shared frozen RLinf prefix", "feature_dim": 4104}
    motor = SimpleNamespace(metadata={"model_sha256": "frozen130"},
                            feature_encoder=SimpleNamespace(contract=features))
    registry = dict(format="rlinf130_trained_arm_v1", completed=True, arm="clean",
                    model_sha256="frozen130", feature_contract=features,
                    noise_contract=NOISE_CONTRACT, collection_cost={"primitive_steps": 3371},
                    models={key: dict(objective=objective, mode=mode,
                                      actual_updates=500, requested_updates=500,
                                      primary={"update": 500, "selection": "final_update"})
                            for objective in ("ad", "dpt") for mode in ("full", "static")
                            for key in (f"{objective}_{mode}",)})
    return registry, motor


def test_episode_lesions_reuse_full_weights_and_static_uses_separate_training():
    for objective in ("ad", "dpt"):
        assert HISTORY_METHODS[f"{objective}_episode"] == (f"{objective}_full", "episode")
        assert HISTORY_METHODS[f"{objective}_static"] == (f"{objective}_static", "static")


@pytest.mark.parametrize("field,value", [
    ("completed", False), ("arm", "pro"), ("model_sha256", "old_openpi"),
    ("feature_contract", {"kind": "teacher-specific encoder"}),
    ("noise_contract", "native full Gaussian"),
])
def test_rejects_incomplete_or_incompatible_corpus_arms(field, value):
    registry, motor = registry_and_motor()
    registry[field] = value
    with pytest.raises(ValueError, match="matching RLinf"):
        validate_registry(registry, motor)


def test_requires_actual_final_update_instead_of_lower_nll_checkpoint():
    registry, motor = registry_and_motor()
    validate_registry(registry, motor)
    bad = copy.deepcopy(registry)
    bad["models"]["ad_full"]["primary"]["update"] = 300
    with pytest.raises(ValueError, match="final-update500"):
        validate_registry(bad, motor)
    validate_registry(bad, motor, students=False)


def test_candidate_hash_and_path_are_bound_to_registry(tmp_path):
    root = tmp_path / "arm"
    root.mkdir()
    path = root / "candidate.pt"
    path.write_bytes(b"verified candidate")
    entry = dict(path="candidate.pt", sha256=sha256(path))
    assert checked_entry(root, entry) == path
    path.write_bytes(b"changed after registry creation")
    with pytest.raises(ValueError, match="artifact hash"):
        checked_entry(root, entry)
    external = tmp_path / "outside.pt"
    external.write_bytes(b"not in this arm")
    with pytest.raises(ValueError, match="registry path"):
        checked_entry(root, dict(path="../outside.pt", sha256=sha256(external)))


def test_closed_loop_retains_actual_feedback_across_episode_reset():
    import numpy as np
    from fluffy_bassoon.meta_rl.vla_history import FrozenHistoryController, HistoryConfig, HistoryNoisePolicy
    from fluffy_bassoon.meta_rl.vla_libero import run_episode

    observation = dict(image=np.zeros((64, 64, 3), np.uint8),
                       wrist=np.zeros((64, 64, 3), np.uint8), state=np.zeros(8, np.float32))

    class Simulator:
        metadata = dict(instruction="basket", task_id=1, suite="development", bddl_file="basket.bddl")

        def reset(self, initial_state, scene_seed):
            return dict(observation=observation, initial_observation_sha256="fixed")

        def step(self, actions):
            return dict(observation=observation, success=True, done=True,
                        executed_actions=actions, rewards=[1.], primitive_steps=1)

    motor = SimpleNamespace(horizon=10,
        infer=lambda obs, instruction, noise: np.zeros((10, 7), np.float32))
    policy = HistoryNoisePolicy(HistoryConfig(4, 32, width=8, layers=1, heads=2, context_length=16))
    controller = FrozenHistoryController(policy, mode="full")
    seen = []

    def encoder(obs, instruction):
        seen.append((obs, instruction))
        return np.zeros(4, np.float32)

    choose, feedback = history_callbacks(controller, encoder, "basket", deterministic=True)
    for _ in range(2):
        rows, result = run_episode(Simulator(), motor, choose, initial_state=0,
            scene_seed=0, chunk=10, record_features=False, feedback=feedback)
        assert result["success"] and controller.previous_done and not controller.pending
        np.testing.assert_array_equal(controller.previous_noise, rows[-1]["executed_noise"])
        assert controller.previous_reward == 0.
    assert len(seen) == 2
    assert controller.state is not None
