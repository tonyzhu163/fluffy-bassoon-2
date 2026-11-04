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


def test_counted_native_factory_does_not_stack_hooks_across_persistent_reconfigures(monkeypatch):
    from robosuite.environments.base import MujocoEnv
    from robosuite.utils.binding_utils import MjSim
    base_step_calls, base_tick_calls = [], []

    def base_step(self, action):
        base_step_calls.append(action)
        self.cur_time += 0.04
        return "step-ok"

    def base_tick(self, *args, **kwargs):
        base_tick_calls.append(1)

    monkeypatch.setattr(MujocoEnv, "step", base_step)
    for name in ("step", "step1", "step2"):
        monkeypatch.setattr(MjSim, name, base_tick)

    for episode in range(3):
        env = SimpleNamespace(control_freq=25., control_timestep=.04, model_timestep=.002,
            lite_physics=False, cur_time=0.)
        env.reset = lambda: "reset-ok"
        wrapped = counted_native_factory(lambda e=env: e, physics_tick_limit=2)()
        wrapped.reset()
        MujocoEnv.step(wrapped, f"act-{episode}")
        MjSim.step(SimpleNamespace())
        MjSim.step(SimpleNamespace())
        assert wrapped._flora_accounting["controls"] == 1
        assert wrapped._flora_accounting["physics_ticks"] == 2
    assert len(base_step_calls) == 3
    assert len(base_tick_calls) == 6


def test_residual_factory_uses_declared_ppo_cap_and_preserves_native_ood(monkeypatch):
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env
    from rlinf.envs.sim.robocasa365.utils import resolve_robocasa365_episode_horizons
    from fluffy_bassoon.meta_rl.robocasa_residual import native_environment_factory
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS, OOD_TASKS, TASKS
    # Exercise the production factory and pinned native resolver without physics.
    captured=[]
    def capture(self, cfg, **kwargs):
        captured.append(cfg)
        self.task_horizons=resolve_robocasa365_episode_horizons(
            task_horizons=[750], max_episode_steps=cfg.max_episode_steps,
            episode_horizon_source=cfg.episode_horizon_source)
    monkeypatch.setattr(Robocasa365Env, '__init__', capture)
    factory=native_environment_factory(OmegaConf.create(dict(episode_horizon_source='task_horizon')))
    for task in (*PPO_TASKS,*OOD_TASKS):
        env=factory(dict(task=task,native_seed=42))
        assert list(env.task_horizons)==[TASKS[task][0]]
        assert captured[-1].max_steps_per_rollout_epoch==TASKS[task][0]
    assert [TASKS[t][0] for t in PPO_TASKS]==[600]*5
    assert [TASKS[t][0] for t in OOD_TASKS]==[900,1050,450,600,450]
