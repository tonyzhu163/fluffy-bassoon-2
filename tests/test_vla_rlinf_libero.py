"""Backbone identity and native observation/noise contracts; allocated Linux."""

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_rlinf_libero import (MODEL_REPOSITORY, MODEL_REVISION,
    MODEL_SHA256, NORM_FILE, environment_observation, model_configuration)


def test_rlinf130_is_distinct_from_official_openpi_checkpoint_and_has_native_eval_config():
    assert MODEL_REPOSITORY == "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT"
    assert MODEL_REVISION == "6222623f635769bfc73c9472e29fab9b7fd8e027"
    assert MODEL_SHA256 == "4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683"
    config = model_configuration("/tmp/released-rlinf130")
    assert config["openpi"]["task"] == "eval" and not config["add_value_head"]
    assert config["openpi"]["action_horizon"] == 10 and config["openpi"]["model_action_dim"] == 32
    assert config["openpi_data"]["norm_stats_path"].endswith(NORM_FILE)
    assert not config["openpi"]["torch_compile"] and not config["openpi"]["discrete_state_input"]


def test_native_rlinf_observation_uses_only_public_images_state_instruction_and_copies_arrays():
    obs = dict(image=np.zeros((256, 256, 3), np.uint8), wrist=np.ones((256, 256, 3), np.uint8),
               state=np.arange(8, dtype=np.float32), privileged_object_state=np.ones(100))
    native = environment_observation(obs, "put the object in the tray")
    assert set(native) == {"main_images", "wrist_images", "states", "task_descriptions"}
    assert native["main_images"].shape == (1, 256, 256, 3) and native["states"].shape == (1, 8)
    native["main_images"][:] = 1
    assert not obs["image"].any()
    with pytest.raises(ValueError, match="eight public"):
        environment_observation(dict(obs, state=np.zeros(9)), "task")
