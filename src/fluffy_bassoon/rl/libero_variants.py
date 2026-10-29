"""Released RLinf pi0.5 LIBERO motor on LIBERO-Plus/PRO variants under fixed noise arms.

outputs/libero-noise-icrl-20261029-r1/PLAN.md, Stage L0 (headroom per perturbation
type), no training. Both packages ship perturbed variants of LIBERO-10 (and of Spatial,
Object, Goal) but not of LIBERO-90, whose entries are the originals; this screen uses
LIBERO-10, the held-out tasks of the frozen split.

- LIBERO-Plus (`rlinf-liberoplus`): every variant is a task of the package's own
  `libero_10` benchmark. Camera, robot-initial-state and sensor-noise variants are
  encoded in the task name (`_view_<h>_<v>_<scale>_<rot>_<vert>_initstate_<k>[_noise_<n>]`),
  which the package's ControlEnv parses; language, light, background (`_table_`,
  `_tb_`), layout (`_add_`) and level/sample variants have their own BDDL files.
- LIBERO-PRO (`rlinf-liberopro`): one BDDL and init file per base task in each
  perturbation directory `libero_10_<kind>`, as RLinf e1a82eb's LiberoEnv reads them.

Model, observation and action conventions follow RLinf e1a82eb's LIBERO pi0.5
evaluation, as on branch `libero-goal-headroom` (src/fluffy_bassoon/rl/libero_goal.py):
256 px views rotated 180 degrees, 8-D state, 5 flow steps, horizon 10 with 5 executed,
15 settle steps with the gripper open. LIBERO-10 episodes run 520 steps (openpi).
"""
from collections import defaultdict
import hashlib
import importlib
import os
from pathlib import Path
import re
import sys

import numpy as np


SUITE = "libero_10"
CHUNK, HORIZON, MODEL_ACTION_DIM, CAMERA, SETTLE, STEPS = 5, 10, 32, 256, 15, 520
NOISE_SHAPE = (HORIZON, MODEL_ACTION_DIM)
NATIVE = -1
PLUS_FILE_MARKERS = (("language", "_language_"), ("light", "_light_"), ("background", "_table_"),
                     ("background", "_tb_"), ("layout", "_add_"), ("level", "_level"))
VIEW = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)(?:_noise_(\d+))?$")
PRO_KINDS = ("lan", "object", "swap", "task")


def plus_category(name):
    """Perturbation type of a LIBERO-Plus task name; 'clean' for an unperturbed base task."""
    for category, marker in PLUS_FILE_MARKERS:
        if marker in name:
            return category
    view = VIEW.search(name)
    if view is None:
        return "clean"
    *angles, scale, rot, vert, init_state, noise = view.groups()
    if noise is not None and int(noise) > 0:
        return "sensor_noise"
    if (int(angles[0]), int(angles[1]), int(scale), int(rot), int(vert)) != (0, 0, 100, 0, 0):
        return "camera"
    if int(init_state) != 0:
        return "robot_init"
    return "clean"


def pro_category(directory):
    kind = directory.removeprefix(f"{SUITE}_")
    if kind.startswith("with_"):
        return "distractor"
    if kind in PRO_KINDS:
        return {"lan": "language", "swap": "position"}.get(kind, kind)
    return None


SCENE = re.compile(r"^[A-Z_]+_SCENE\d+_")


def base_task(name, bases):
    """The LIBERO-10 base task a variant name extends (longest matching prefix).

    LIBERO-PRO renames the scene of some variants, so a name whose scene differs
    matches on the task part after the scene prefix.
    """
    def extends(text, base):
        return text == base or text.startswith(base + "_") or text.startswith(base + " ")
    matches = [b for b in bases if extends(name, b)]
    if not matches:
        stripped = SCENE.sub("", name)
        matches = [b for b in bases if extends(stripped, SCENE.sub("", b))]
    if not matches:
        raise ValueError(f"No base task for {name}")
    return max(matches, key=len)


def select(variants, per_category, seed):
    """Up to `per_category` variants per category, spread over base tasks, seeded."""
    rng = np.random.default_rng(seed)
    by_category = defaultdict(lambda: defaultdict(list))
    for variant in sorted(variants, key=lambda v: v["name"]):
        by_category[variant["category"]][variant["base_task"]].append(variant)
    chosen = []
    for category in sorted(by_category):
        groups = by_category[category]
        bases = sorted(groups)
        queues = [[groups[b][i] for i in rng.permutation(len(groups[b]))] for b in
                  (bases[i] for i in rng.permutation(len(bases)))]
        picked = []
        while len(picked) < per_category and any(queues):
            for queue in queues:
                if queue and len(picked) < per_category:
                    picked.append(queue.pop(0))
        chosen += [dict(v) for v in picked]
    for index, variant in enumerate(chosen):
        variant["variant"] = index
    return chosen


def arm_noise(seeds):
    """One fixed initial-noise array per arm seed, [arms, *NOISE_SHAPE] float32."""
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("Require integer arm seeds in [0, 2**63)")
    return np.stack([np.random.default_rng(seed).standard_normal(NOISE_SHAPE).astype(np.float32)
                     for seed in seeds])


def native_noise(seed, chunk):
    return np.random.default_rng([seed, chunk]).standard_normal(NOISE_SHAPE).astype(np.float32)


def env_seed(scene):
    """Fixture-placement seed shared by every arm on one scene."""
    return 1000 + scene


# ---------------------------------------------------------------- packages

def route(kind):
    """Make `libero.libero` resolve to the Plus/PRO package, as RLinf's LiberoEnv does."""
    if kind not in ("plus", "pro"):
        raise ValueError("Kind is 'plus' or 'pro'")
    os.environ["LIBERO_TYPE"] = kind
    package = f"libero{kind}"
    core = importlib.import_module(f"{package}.{package}")
    sys.modules["libero"] = importlib.import_module(package)
    sys.modules["libero.libero"] = core
    sys.modules["libero.libero.benchmark"] = importlib.import_module(f"{package}.{package}.benchmark")
    sys.modules["libero.libero.envs"] = envs = importlib.import_module(f"{package}.{package}.envs")
    return core, envs


def bddl_language(path):
    text = Path(path).read_text()
    match = re.search(r"\(:language\s+([^)]*)\)", text)
    return " ".join(match.group(1).split()) if match else None


def held_out_bases():
    """LIBERO-10 base task names: the frozen split's test tasks."""
    from fluffy_bassoon import libero_split
    tasks = libero_split.tasks(libero_split.load(), "test")
    if any(task["suite"] != SUITE for task in tasks):
        raise ValueError("Split test tasks must be LIBERO-10")
    return sorted(task["semantic_task"] for task in tasks)


def enumerate_variants(kind):
    """All LIBERO-10 variants of one package with category, base task, BDDL and language.

    'clean' is each base task's original BDDL and initial states inside the package.
    """
    core, _ = route(kind)
    benchmark = sys.modules["libero.libero.benchmark"]
    bddl_root = Path(core.get_libero_path("bddl_files"))
    init_root = Path(core.get_libero_path("init_states"))
    bases = held_out_bases()
    variants = [dict(kind=kind, category="clean", base_task=name, name=name,
                     language=bddl_language(bddl_root/SUITE/f"{name}.bddl"),
                     bddl=str(bddl_root/SUITE/f"{name}.bddl"),
                     init_states=str(init_root/SUITE/f"{name}.pruned_init")) for name in bases]
    if kind == "plus":
        suite = benchmark.get_benchmark(SUITE)()
        for index, name in enumerate(suite.get_task_names()):
            variants.append(dict(kind=kind, category=plus_category(name), base_task=base_task(name, bases),
                                 name=name, task_index=index, language=suite.get_task(index).language,
                                 bddl=suite.get_task_bddl_file_path(index)))
        return variants
    for directory in sorted(p.name for p in bddl_root.iterdir() if p.is_dir() and p.name.startswith(SUITE + "_")):
        category = pro_category(directory)
        if category is None:
            continue
        for path in sorted((bddl_root/directory).glob("*.bddl")):
            init = init_root/directory/f"{path.stem}.pruned_init"
            if init.exists():
                variants.append(dict(kind=kind, category=category, base_task=base_task(path.stem, bases),
                                     name=f"{directory}/{path.stem}", directory=directory,
                                     language=bddl_language(path), bddl=str(path), init_states=str(init)))
    return variants


def init_states(variant):
    import torch
    if "init_states" in variant:
        return torch.load(variant["init_states"], weights_only=False)
    benchmark = sys.modules["libero.libero.benchmark"]
    return benchmark.get_benchmark(SUITE)().get_task_init_states(variant["task_index"])


# ---------------------------------------------------------------- workers

def policy_obs(obs):
    from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle
    return dict(full_image=np.ascontiguousarray(get_libero_image(obs)),
                wrist_image=np.ascontiguousarray(get_libero_wrist_image(obs)),
                state=np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                                      obs["robot0_gripper_qpos"]]))


def _worker(conn, kind):
    env, built = None, None
    try:
        _, envs = route(kind)
        while True:
            command, payload = conn.recv()
            if command == "close":
                break
            if command == "reset":
                if built != payload["bddl"]:
                    if env is not None:
                        env.close()
                    env = envs.OffScreenRenderEnv(bddl_file_name=payload["bddl"], camera_heights=CAMERA,
                                                  camera_widths=CAMERA)
                    built = payload["bddl"]
                env.seed(payload["seed"])
                env.reset()
                env.set_init_state(payload["init_state"])
                settle = np.zeros(7)
                settle[-1] = -1
                for _ in range(SETTLE):
                    obs, _, done, _ = env.step(settle)
                state = hashlib.sha256(np.ascontiguousarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
                conn.send(("ok", dict(obs=policy_obs(obs), done=bool(done), settled_state_sha256=state,
                                      pid=os.getpid())))
            elif command == "step":
                done = False
                for action in payload:
                    obs, _, done, _ = env.step(np.asarray(action, dtype=np.float64))
                    if done:
                        break
                conn.send(("ok", dict(obs=policy_obs(obs), done=bool(done))))
            else:
                raise ValueError(command)
    except Exception as error:  # report to the driver instead of hanging it
        import traceback
        conn.send(("error", traceback.format_exc() + repr(error)))
    finally:
        if env is not None:
            env.close()


class EnvPool:
    """One LIBERO-Plus/PRO OffScreenRenderEnv per spawned worker process."""

    def __init__(self, size, kind):
        import multiprocessing as mp
        context = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(size):
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, args=(child, kind), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def _call(self, command, payloads):
        conns = self.conns[:len(payloads)]
        for conn, payload in zip(conns, payloads):
            conn.send((command, payload))
        out = []
        for conn in conns:
            status, value = conn.recv()
            if status != "ok":
                raise RuntimeError(f"LIBERO worker failed:\n{value}")
            out.append(value)
        return out

    def reset(self, payloads):
        return self._call("reset", payloads)

    def step(self, chunks):
        return self._call("step", chunks)

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=30)

    @property
    def pids(self):
        return [p.pid for p in self.procs]


# ---------------------------------------------------------------- model

def model_config(model_path, norm_stats_path):
    """RLinf e1a82eb pi0_5 model defaults with its LIBERO evaluation overrides."""
    from omegaconf import OmegaConf
    return OmegaConf.create(dict(
        model_type="openpi", model_path=str(model_path), precision=None, pi05=True,
        is_lora=False, num_action_chunks=CHUNK, action_dim=7, use_proprio=True, num_steps=5,
        add_value_head=False, openpi_data=dict(norm_stats_path=str(norm_stats_path)),
        openpi=dict(task="eval", config_name="pi05_libero", num_images_in_input=2, noise_level=.3,
                    action_chunk=CHUNK, num_steps=5, train_expert_only=True, action_env_dim=7,
                    noise_method="flow_sde", add_value_head=False, value_after_vlm=False,
                    value_vlm_mode="mean_token", detach_critic_input=False,
                    model_action_dim=MODEL_ACTION_DIM, paligemma_variant="gemma_2b",
                    action_expert_variant="gemma_300m", max_token_len=200, torch_compile=False)))


def model_obs(states, prompts):
    import torch
    from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
    packed = to_tensor(list_of_dict_to_dict_of_list([s["obs"] for s in states]))
    return dict(main_images=torch.stack([v.clone() for v in packed["full_image"]]),
                wrist_images=torch.stack([v.clone() for v in packed["wrist_image"]]),
                states=packed["state"], task_descriptions=list(prompts))


def run_batch(predict, pool, rows, *, arms, steps=STEPS):
    """Synchronized episodes for one batch of rows; `predict(states, prompts, noise)` -> [B, CHUNK, 7].

    Arm k passes arms[k] as the initial noise of every chunk; arm -1 draws fresh noise per
    chunk from a generator seeded by the row's episode seed. Finished envs stop stepping.
    """
    states = pool.reset([dict(bddl=r["bddl"], seed=env_seed(r["scene_index"]), init_state=r["init_state"])
                         for r in rows])
    first = [dict(settled_state_sha256=s["settled_state_sha256"], worker_pid=s["pid"]) for s in states]
    done = [s["done"] for s in states]
    issued = [[] for _ in rows]
    prompts = [r["language"] for r in rows]
    for chunk in range(-(-steps // CHUNK)):
        live = [i for i in range(len(rows)) if not done[i]]
        if not live:
            break
        noise = np.stack([native_noise(rows[i]["episode_seed"], chunk) if rows[i]["arm"] == NATIVE
                          else arms[rows[i]["arm"]] for i in live])
        actions = np.asarray(predict([states[i] for i in live], [prompts[i] for i in live], noise))
        if actions.shape != (len(live), CHUNK, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"Unexpected action chunk {actions.shape}")
        results = pool.step([a for a in actions])
        for i, action, result in zip(live, actions, results):
            states[i] = result
            done[i] = result["done"]
            issued[i].append(action.astype(np.float32))
    out = []
    for row, info, success, chunks in zip(rows, first, done, issued):
        stacked = np.stack(chunks) if chunks else np.zeros((0, CHUNK, 7), np.float32)
        out.append((dict({k: v for k, v in row.items() if k not in ("init_state", "bddl")}, success_once=bool(success),
                         chunks=len(chunks), issued_actions_sha256=hashlib.sha256(stacked.tobytes()).hexdigest(),
                         slot=0, **info), stacked[:1]))
    return out
