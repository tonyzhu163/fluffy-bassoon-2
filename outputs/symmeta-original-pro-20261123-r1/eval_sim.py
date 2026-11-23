"""One fresh original-LIBERO-PRO Panda episode in the simulator interpreter."""
from hashlib import sha256
from multiprocessing.connection import Connection
import os
from pathlib import Path
import random
import re
import sys
import traceback

import numpy as np


def digest(a):
    a = np.ascontiguousarray(a)
    return sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def public(obs):
    # LIBERO returns xyzw. The pinned OpenPI client calls the same field *_quat_xyzw.
    fields = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_gripper_qpos")
    raw = {k: np.asarray(obs[k]).copy() for k in fields}
    raw["robot0_eef_quat_xyzw"] = np.asarray(obs["robot0_eef_quat"]).copy()
    for key in fields[:2]:
        if raw[key].shape != (256, 256, 3) or raw[key].dtype != np.uint8:
            raise ValueError(f"Invalid policy camera {key}: {raw[key].shape} {raw[key].dtype}")
    return raw


def checked_file(root, relative, expected_sha):
    path = root / relative
    if sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Released asset digest mismatch: {path}")
    return path


def camera_receipt(env):
    import mujoco

    # Replay the released scene's camera declarations into a recorder, without editing the model.
    declared = {}

    class Recorder:
        def set_camera(self, camera_name, pos, quat, **kwargs):
            declared[camera_name] = (pos, quat)

    env.env._setup_camera(Recorder())
    receipt = {}
    for name in ("agentview", "robot0_eye_in_hand"):
        idx = env.sim.model.camera_name2id(name)
        xyz = env.sim.data.cam_xpos[idx]
        rotation = env.sim.data.cam_xmat[idx].reshape(-1)
        receipt[name] = dict(xyz=xyz.tolist(), world_matrix=rotation.tolist(),
                             fovy=float(env.sim.model.cam_fovy[idx]))
        if name == "agentview":
            position, quat = declared[name]
            expected = np.empty(9)
            quat = np.asarray(quat, np.float64)
            mujoco.mju_quat2Mat(expected, quat / np.linalg.norm(quat))
            receipt[name]["declared_xyz"] = list(position)
            receipt[name]["declared_wxyz"] = quat.tolist()
            # Compare at the precision of the source's float32 camera parameters.
            tol = np.finfo(np.float32).eps
            if not (np.allclose(xyz, position, atol=tol, rtol=tol)
                    and np.allclose(rotation, expected, atol=tol, rtol=tol)):
                raise ValueError(f"Native agentview pose differs from released scene: {receipt}")
    return receipt


def create(case, seed, cpu):
    import inspect
    import torch
    from OpenGL import GL
    from liberopro.liberopro.envs import OffScreenRenderEnv
    from liberopro.liberopro.benchmark import grab_language_from_filename

    release = Path(os.environ["ORIGINAL_PRO_ROOT"]).resolve()
    origin = Path(inspect.getfile(OffScreenRenderEnv)).resolve()
    if release not in origin.parents:
        raise ValueError(f"Wrong simulator import: {origin}")
    assets = Path(os.environ["ORIGINAL_PRO_ASSETS"])
    bddl = checked_file(assets, case["bddl_path"], case["bddl_sha256"])
    init = checked_file(assets, case["init_path"], case["init_sha256"])
    # The OpenPI client passes task.language from the benchmark registry.
    source = re.sub(r";[^\n]*", "", bddl.read_text(encoding="utf-8"))
    language, = re.findall(r"\(\s*:language\s+([^()]*)\)", source, re.I)
    bddl_prompt = " ".join(language.split())
    prompt = grab_language_from_filename(bddl.name)
    if prompt != case["policy_prompt"] or bddl_prompt != case["bddl_prompt"]:
        raise ValueError(f"Manifest/BDDL instruction mismatch in {bddl}")
    states = torch.load(init, map_location="cpu", weights_only=False)
    if list(np.shape(states)) != case["init_shape"]:
        raise ValueError(f"Released state shape mismatch in {init}")
    state = np.asarray(states[case["init_index"]])
    state_hashes = [digest(row) for row in states]
    random.seed(seed)
    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(state)
    arm = env.robots[0]
    robot = type(arm.robot_model).__name__
    controller = type(arm.controller).__name__
    low, high = env.env.action_spec
    mounted_scenes = ("libero_tabletop_manipulation", "libero_kitchen_tabletop_manipulation")
    native_robot = "MountedPanda" if env.problem_name.lower() in mounted_scenes else "OnTheGroundPanda"
    if (len(env.robots) != 1 or robot != native_robot or arm.controller.name != "OSC_POSE"
            or env.env.control_freq != 20 or np.shape(low) != (7,) or np.shape(high) != (7,)):
        raise ValueError(f"Unknown action interface: {robot}/{controller}/{arm.controller.name}/{low}/{high}")
    if env.language_instruction != bddl_prompt.lower() or env.problem_name.lower() != case["problem"].lower():
        raise ValueError(f"BDDL parse mismatch in {bddl}: {env.language_instruction!r}")
    env.sim._render_context_offscreen.gl_ctx.make_current()
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    if not cpu and "nvidia" not in vendor.lower():
        raise ValueError(f"Expected allocated NVIDIA renderer, got {vendor}/{renderer}")
    if not np.array_equal(env.get_sim_state(), state):
        raise ValueError("Original simulator failed to restore the selected state exactly")
    receipt = dict(initial_cameras=camera_receipt(env), case_id=case["case_id"], seed=seed, init_index=case["init_index"],
                   bddl_sha256=case["bddl_sha256"], init_sha256=case["init_sha256"],
                   selected_state=digest(state), bank_unique_states=len(set(state_hashes)),
                   selected_state_multiplicity=state_hashes.count(digest(state)), initial_physics=digest(env.get_sim_state()),
                   initial_public={k: digest(v) for k, v in public(obs).items()},
                   initial_success=bool(env.check_success()), robot=robot, controller=controller,
                   controller_name=arm.controller.name, control_freq=env.env.control_freq,
                   low=low.tolist(), high=high.tolist(), vendor=vendor, renderer=renderer,
                   prompt=prompt, bddl_prompt=bddl_prompt, parsed_instruction=env.language_instruction, problem=env.problem_name,
                   max_steps=case["max_steps"], settle_rewards=[], settle_done=[])
    # Match OpenPI's ten-step settling phase, including its handling of a transient done.
    for _ in range(10):
        obs, reward, done, _ = env.step([0.] * 6 + [-1.])
        receipt["settle_rewards"].append(float(reward))
        receipt["settle_done"].append(bool(done))
    receipt["settled_cameras"] = camera_receipt(env)
    receipt["settled_physics"] = digest(env.get_sim_state())
    return env, (receipt, public(obs))


def serve(fd):
    sys.path.insert(0, os.environ["ORIGINAL_PRO_ROOT"])
    env = None
    controls = 0
    with Connection(fd) as pipe:
        try:
            op, value = pipe.recv()
            if op != "create":
                raise ValueError("Expected create")
            case, seed, cpu = value
            env, initial = create(case, seed, cpu)
            pipe.send((True, initial))
            cap = case["max_steps"]
            while controls < cap:
                op, actions = pipe.recv()
                if op != "chunk" or actions.shape != (min(5, cap - controls), 7):
                    raise ValueError("Expected one five-control policy chunk")
                if not np.isfinite(actions).all():
                    raise ValueError("Non-finite action")
                rewards = []
                for action in actions:
                    # The released client passes the proposal through to the Panda controller.
                    obs, reward, done, _ = env.step(action.tolist())
                    controls += 1
                    rewards.append(float(reward))
                    if done:
                        break
                pipe.send((True, dict(observation=public(obs), rewards=rewards,
                                      done=bool(done), physics=digest(env.get_sim_state()))))
                if done:
                    break
        except EOFError:
            pass
        except Exception:
            pipe.send((False, traceback.format_exc()))
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
