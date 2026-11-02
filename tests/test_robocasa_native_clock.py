"""Native lifecycle numerical check; imports classes but constructs no simulator."""
from types import SimpleNamespace

from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo_env import counted_native_factory


def test_clock_is_read_from_initialized_native_reset_not_constructor(monkeypatch):
    from robosuite.environments.base import MujocoEnv
    from robosuite.utils.binding_utils import MjSim
    # Restore the class hooks after this isolated test; no real step is run.
    for cls, names in ((MujocoEnv, ("step",)), (MjSim, ("step", "step1", "step2"))):
        for name in names:
            monkeypatch.setattr(cls, name, getattr(cls, name))
    env = SimpleNamespace(control_freq=25., control_timestep=None, model_timestep=None,
        lite_physics=True, cur_time=None)

    def reset():
        env.control_timestep, env.model_timestep, env.cur_time = .04, .002, 0.
        return "actual-reset"

    env.reset = reset
    actual = counted_native_factory(lambda: env, physics_tick_limit=100_000)()
    assert actual._flora_accounting["clock"] is None
    assert actual.reset() == "actual-reset"
    assert actual._flora_accounting["clock"] == dict(control_freq=25., control_timestep=.04, model_timestep=.002, lite_physics=True)
    assert actual._flora_accounting["phase"] == "policy"
    assert actual._flora_accounting["control_time"] == 0.
