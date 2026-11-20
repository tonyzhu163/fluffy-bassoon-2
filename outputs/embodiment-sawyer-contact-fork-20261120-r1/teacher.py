"""One source-qualified Sawyer rim push for both branches of the contact fork."""


def make_teacher(env, obs):
    """Bind a fresh controller to this ArmEnv at its current native control boundary."""
    # The runner sets the pinned evaluator import paths before opening ArmEnv.
    import mujoco
    from recover_teacher import paddle_handoff
    from robosuite.utils.binding_utils import MjData
    from sim import _PushGeometry

    if (env.arm, env.suite, env.task_id) != ("Sawyer", "libero_goal", 5):
        raise ValueError("The fixed teacher requires Sawyer on native Goal 5 BASE")
    wrapper = env._env
    if wrapper is None or env._done:
        raise ValueError("Construct the teacher on an open, nonterminal episode")

    geometry = _PushGeometry(wrapper)
    live = wrapper.sim.data._data
    geometry.data = MjData(geometry.model)
    scratch = geometry.data._data

    def current_scene():
        # Euler integration advances qpos beyond the most recent position cache.
        # Rebuild the observed endpoint in scratch without forwarding the live OSC.
        scratch.qpos[:] = live.qpos
        scratch.mocap_pos[:] = live.mocap_pos
        scratch.mocap_quat[:] = live.mocap_quat
        scratch.eq_active[:] = live.eq_active
        scratch.time = live.time
        mujoco.mj_fwdPosition(geometry.model._model, scratch)
        return geometry.read()

    # Candidate zero in R2 sweep_bank.candidates(seed, 0, ...), for every seed.
    params = dict(mode="rim", tilt=28, yaw=180, flank=-35, bottom=0.006)
    command, alignment = paddle_handoff(params, current_scene(), geometry)
    expected_step = env._step
    horizon = min(300, env.entry["max_steps"])

    def act(obs):
        nonlocal expected_step
        if env._env is not wrapper or env._done or env._step >= horizon:
            raise RuntimeError("No remaining controls on this teacher's episode")
        if env._step != expected_step:
            raise RuntimeError("Request exactly one teacher action per issued native control")
        action, _ = command(obs, current_scene(), geometry)
        expected_step += 1
        return action

    act.handoff = alignment
    return act
