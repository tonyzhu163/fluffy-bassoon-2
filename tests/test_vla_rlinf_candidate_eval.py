import copy
from types import SimpleNamespace

import pytest

from fluffy_bassoon.meta_rl.vla_libero import sha256
from fluffy_bassoon.meta_rl.vla_rlinf_candidate_eval import (
    HISTORY_METHODS, NOISE_CONTRACT, checked_entry, validate_registry,
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
