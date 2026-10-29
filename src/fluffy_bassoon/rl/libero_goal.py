"""Frozen RLinf pi0.5 on LIBERO-Goal: instruction-steering headroom check.

All ten LIBERO-Goal tasks share one scene definition (their BDDL files differ only in
language, objects of interest and goal), so an init state from any task can be loaded
into any task's environment. Each episode records all ten goal predicates at every
step, which shows which goal the policy actually pursued, not only the instructed one.

Model, observation and action conventions follow RLinf e1a82eb's LIBERO evaluation
(``pi05_libero``, 5-step chunks of a 10-step horizon, 5 Euler steps, 256 px cameras
rotated 180 degrees, 15 settle steps with the gripper open, 320 environment steps).
Initial flow noise is drawn per (init state, chunk), so instruction conditions on the
same init state share their noise regardless of batching.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


SUITE = "libero_goal"
CHUNK, HORIZON, STEPS, SETTLE = 5, 10, 320, 15
MODEL_ACTION_DIM, CAMERA = 32, 256
VAGUE_PROMPTS = ("complete the task", "do the correct task")
# LIBERO benchmark order (libero.libero.benchmark task_maps["libero_goal"]); checked at runtime.
EXPECTED_TASKS = (
    "open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet", "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet", "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl", "turn_on_the_stove", "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack")
CRITERIA = dict(explicit_min=.8, swap_follow_min=.8, gap_min=.3)


# ---------------------------------------------------------------- plan

def noise_seed(base, source_task, trial, chunk):
    """Seed shared by every instruction run on the same (source task, trial) init state."""
    if not (0 <= source_task < 100 and 0 <= trial < 100 and 0 <= chunk < 100):
        raise ValueError("Seed components out of range")
    return int(base) * 1_000_000 + source_task * 10_000 + trial * 100 + chunk


def build_plan(n_tasks=10, explicit_trials=12, paired_trials=2, vague_prompts=VAGUE_PROMPTS):
    """Ordered episode plan; earlier phases are the most informative if time runs out.

    Phase order: explicit on the paired trials, vague, swap, remaining explicit trials.
    Paired trials (0..paired_trials-1) of each source task are run under every
    instruction (own = explicit, others = swap) and every vague prompt.
    """
    if not 0 < paired_trials <= explicit_trials:
        raise ValueError("Paired trials must be a prefix of the explicit trials")
    plan = []

    def add(condition, source, trial, instructed=None, prompt=None):
        plan.append(dict(condition=condition, source_task=source, trial=trial,
                         instructed_goal=instructed, vague_prompt=prompt))

    for trial in range(paired_trials):
        for task in range(n_tasks):
            add("explicit", task, trial, instructed=task)
    for prompt in vague_prompts:
        for trial in range(paired_trials):
            for task in range(n_tasks):
                add("vague", task, trial, prompt=prompt)
    for trial in range(paired_trials):
        for source in range(n_tasks):
            for goal in range(n_tasks):
                if goal != source:
                    add("swap", source, trial, instructed=goal)
    for trial in range(paired_trials, explicit_trials):
        for task in range(n_tasks):
            add("explicit", task, trial, instructed=task)
    for index, row in enumerate(plan):
        row["episode"] = index
    return plan


def env_seed(row):
    """LIBERO fixture-placement seed shared by every condition on one init state."""
    return 1000 + 100 * row["source_task"] + row["trial"]


def instruction(row, languages):
    return row["vague_prompt"] if row["condition"] == "vague" else languages[row["instructed_goal"]]


def chunk_noise(rows, chunk, base_seed, device=None):
    import torch
    noise = [torch.randn((HORIZON, MODEL_ACTION_DIM), dtype=torch.float32,
                         generator=torch.Generator().manual_seed(
                             noise_seed(base_seed, r["source_task"], r["trial"], chunk)))
             for r in rows]
    stacked = torch.stack(noise)
    return stacked if device is None else stacked.to(device)


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


def model_fingerprint(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- LIBERO

def suite_tasks():
    """Benchmark tasks with BDDL paths, languages and parsed goal states."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    suite = benchmark.get_benchmark_dict()[SUITE]()
    tasks = []
    for index in range(suite.get_num_tasks()):
        task = suite.get_task(index)
        path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        goal = robosuite_parse_problem(path)["goal_state"]
        tasks.append(dict(index=index, name=task.name, language=task.language, bddl=path,
                          bddl_sha256=sha256_file(path), goal_state=goal))
    if tuple(t["name"] for t in tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"Unexpected LIBERO-Goal task order: {[t['name'] for t in tasks]}")
    return suite, tasks


def goal_vector(env, goal_states):
    """All goal predicates on the current sim state (LIBERO's own _eval_predicate)."""
    domain = env.env
    return [all(bool(domain._eval_predicate(state)) for state in goal) for goal in goal_states]


def make_env(bddl, render=True):
    if render:
        from libero.libero.envs import OffScreenRenderEnv
        return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAMERA, camera_widths=CAMERA)
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False,
                      use_camera_obs=False, camera_heights=CAMERA, camera_widths=CAMERA)


def policy_obs(obs):
    from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle
    return dict(full_image=np.ascontiguousarray(get_libero_image(obs)),
                wrist_image=np.ascontiguousarray(get_libero_wrist_image(obs)),
                state=np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                                      obs["robot0_gripper_qpos"]]))


def _worker(conn, goal_states):
    env, current = None, None
    try:
        while True:
            command, payload = conn.recv()
            if command == "close":
                break
            if command == "reset":
                if payload["bddl"] != current:
                    if env is not None:
                        env.close()
                    env, current = make_env(payload["bddl"]), payload["bddl"]
                # Seeding before reset fixes LIBERO's fixture placement (re-sampled on every
                # hard reset); identical across all ten tasks for the same seed.
                env.seed(payload["seed"])
                env.reset()
                xml_sha = hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
                env.set_init_state(payload["init_state"])
                settle = np.zeros(7)
                settle[-1] = -1
                for _ in range(SETTLE):
                    obs, _, done, _ = env.step(settle)
                state_sha = hashlib.sha256(np.ascontiguousarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
                conn.send(("ok", dict(obs=policy_obs(obs), goals=goal_vector(env, goal_states),
                                      native_success=bool(done), pid=os.getpid(), model_xml_sha256=xml_sha,
                                      settled_state_sha256=state_sha)))
            elif command == "step":
                goals, native = [], []
                for action in payload:
                    obs, _, done, _ = env.step(np.asarray(action, dtype=np.float64))
                    goals.append(goal_vector(env, goal_states))
                    native.append(bool(done))
                conn.send(("ok", dict(obs=policy_obs(obs), goals=goals, native_success=native)))
            else:
                raise ValueError(command)
    except Exception as error:  # report to the driver instead of hanging it
        import traceback
        conn.send(("error", traceback.format_exc() + repr(error)))
    finally:
        if env is not None:
            env.close()


class EnvPool:
    """One LIBERO OffScreenRenderEnv per spawned worker process."""

    def __init__(self, size, goal_states):
        import multiprocessing as mp
        context = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(size):
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, args=(child, goal_states), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def _gather(self, conns):
        out = []
        for conn in conns:
            status, value = conn.recv()
            if status != "ok":
                raise RuntimeError(f"LIBERO worker failed:\n{value}")
            out.append(value)
        return out

    def reset(self, payloads):
        conns = self.conns[:len(payloads)]
        for conn, payload in zip(conns, payloads):
            conn.send(("reset", payload))
        return self._gather(conns)

    def step(self, chunks):
        conns = self.conns[:len(chunks)]
        for conn, chunk in zip(conns, chunks):
            conn.send(("step", chunk))
        return self._gather(conns)

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


def model_obs(states, prompts):
    import torch
    from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
    packed = to_tensor(list_of_dict_to_dict_of_list([s["obs"] for s in states]))
    return dict(main_images=torch.stack([v.clone() for v in packed["full_image"]]),
                wrist_images=torch.stack([v.clone() for v in packed["wrist_image"]]),
                states=packed["state"], task_descriptions=list(prompts))


class GoalTrace:
    """Per-episode first-achieved step, end state and native-success parity of all goals."""

    def __init__(self, n_goals, initial, source_task, native_initial):
        self.n, self.step, self.source = n_goals, 0, source_task
        self.initial = list(initial)
        self.first = [-1] * n_goals
        self.last = list(initial)
        self.parity_mismatches = int(bool(initial[source_task]) != bool(native_initial))

    def update(self, goals, native):
        if len(goals) != self.n:
            raise ValueError("Goal vector length changed")
        self.step += 1
        for g, value in enumerate(goals):
            if value and self.first[g] < 0:
                self.first[g] = self.step
        self.last = list(goals)
        self.parity_mismatches += int(bool(goals[self.source]) != bool(native))

    def row(self):
        achieved = [g for g in range(self.n) if self.first[g] >= 0]
        earliest = min((self.first[g] for g in achieved), default=-1)
        return dict(steps=self.step, goals_initial=[bool(v) for v in self.initial],
                    goals_first_step=self.first, goals_at_end=[bool(v) for v in self.last],
                    achieved_goals=achieved,
                    first_goals=[g for g in achieved if self.first[g] == earliest],
                    native_parity_mismatches=self.parity_mismatches)


def run_batch(model, pool, rows, tasks, suite, *, base_seed, device):
    """Synchronized 320-step episodes for one batch of plan rows."""
    import torch
    languages = [t["language"] for t in tasks]
    goal_count = len(tasks)
    payloads = [dict(bddl=tasks[r["source_task"]]["bddl"], seed=env_seed(r),
                     init_state=suite.get_task_init_states(r["source_task"])[r["trial"]]) for r in rows]
    states = initial = pool.reset(payloads)
    prompts = [instruction(r, languages) for r in rows]
    traces = [GoalTrace(goal_count, s["goals"], r["source_task"], s["native_success"])
              for s, r in zip(states, rows)]
    for chunk in range(STEPS // CHUNK):
        noise = chunk_noise(rows, chunk, base_seed, device)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(model_obs(states, prompts), mode="eval", noise=noise)
        actions = actions.detach().float().cpu().numpy()
        if actions.shape != (len(rows), CHUNK, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"Unexpected action chunk {actions.shape}")
        states = pool.step([a for a in actions])
        for trace, state in zip(traces, states):
            for goals, native in zip(state["goals"], state["native_success"]):
                trace.update(goals, native)
    out = []
    for r, prompt, trace, first in zip(rows, prompts, traces, initial):
        rec = dict(r, instruction=prompt, env_seed=env_seed(r), model_xml_sha256=first["model_xml_sha256"],
                   settled_state_sha256=first["settled_state_sha256"], **trace.row())
        g = r["instructed_goal"]
        rec["instructed_success"] = None if g is None else rec["goals_first_step"][g] >= 0
        out.append(rec)
    return out


# ---------------------------------------------------------------- metrics

def wilson(successes, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0., centre - half), min(1., centre + half)]


def rate(successes, n):
    return dict(k=int(successes), n=int(n), rate=(successes / n if n else float("nan")),
                wilson95=wilson(successes, n))


def summarize(records, n_goals=10, vague_prompts=VAGUE_PROMPTS):
    explicit = [r for r in records if r["condition"] == "explicit"]
    swap = [r for r in records if r["condition"] == "swap"]
    vague = [r for r in records if r["condition"] == "vague"]
    achieved = lambda r, g: r["goals_first_step"][g] >= 0
    first_is = lambda r, g: g in r["first_goals"]

    out = dict(episodes=dict(total=len(records), explicit=len(explicit), swap=len(swap), vague=len(vague)))
    out["explicit"] = {g: rate(sum(achieved(r, g) for r in rows), len(rows))
                       for g in range(n_goals) for rows in [[r for r in explicit if r["instructed_goal"] == g]]}
    out["explicit_overall"] = rate(sum(achieved(r, r["instructed_goal"]) for r in explicit), len(explicit))

    by_goal, pairs, matrix = {}, {}, [[0] * (n_goals + 1) for _ in range(n_goals)]
    for g in range(n_goals):
        rows = [r for r in swap if r["instructed_goal"] == g]
        by_goal[g] = dict(follow_any=rate(sum(achieved(r, g) for r in rows), len(rows)),
                          follow_first=rate(sum(first_is(r, g) for r in rows), len(rows)),
                          source_goal_instead=rate(sum(achieved(r, r["source_task"]) and not achieved(r, g)
                                                       for r in rows), len(rows)))
    for r in swap:
        key = f"{r['source_task']}->{r['instructed_goal']}"
        cell = pairs.setdefault(key, dict(n=0, follow_any=0, source_goal=0))
        cell["n"] += 1
        cell["follow_any"] += achieved(r, r["instructed_goal"])
        cell["source_goal"] += achieved(r, r["source_task"])
        for g in r["achieved_goals"]:
            matrix[r["instructed_goal"]][g] += 1
        if not r["achieved_goals"]:
            matrix[r["instructed_goal"]][n_goals] += 1
    out["swap"] = dict(
        overall_follow_any=rate(sum(achieved(r, r["instructed_goal"]) for r in swap), len(swap)),
        overall_follow_first=rate(sum(first_is(r, r["instructed_goal"]) for r in swap), len(swap)),
        by_instructed_goal=by_goal, by_pair=pairs,
        achieved_matrix=dict(rows="instructed goal", columns="achieved goal (any step); last column none",
                             counts=matrix))

    out["vague"] = {}
    for name, rows in [("pooled", vague)] + [(p, [r for r in vague if r["vague_prompt"] == p]) for p in vague_prompts]:
        first = [0] * (n_goals + 1)
        for r in rows:
            for g in (r["first_goals"] or [n_goals]):
                first[g] += 1
        out["vague"][name] = dict(
            n=len(rows), achieved_any={g: rate(sum(achieved(r, g) for r in rows), len(rows)) for g in range(n_goals)},
            any_goal=rate(sum(bool(r["achieved_goals"]) for r in rows), len(rows)),
            first_goal_counts=first)

    pooled = out["vague"]["pooled"]["achieved_any"]
    out["gap"] = {g: out["explicit"][g]["rate"] - pooled[g]["rate"] for g in range(n_goals)}
    # Goals reached when not instructed (swap + vague): candidate "default" behaviours.
    uninstructed = swap + vague
    out["uninstructed_goal_rate"] = {
        g: rate(sum(achieved(r, g) for r in rows), len(rows))
        for g in range(n_goals) for rows in [[r for r in uninstructed if r["instructed_goal"] != g]]}
    out["checks"] = dict(
        native_parity_mismatch_steps=sum(r["native_parity_mismatches"] for r in records),
        episodes_with_goal_true_after_settle=sum(any(r["goals_initial"]) for r in records),
        incomplete_episodes=sum(r["steps"] != STEPS for r in records),
        init_states_with_unpaired_start=unpaired_starts(records))
    out["criteria"] = criteria(out, n_goals)
    return out


def unpaired_starts(records):
    """(source, trial) init states whose conditions did not start from one identical sim state."""
    starts = {}
    for r in records:
        key = (r["source_task"], r["trial"])
        starts.setdefault(key, set()).add((r.get("model_xml_sha256"), r.get("settled_state_sha256")))
    return sum(len(v) > 1 for v in starts.values())


def criteria(summary, n_goals=10, k=None):
    """Provisional pass criteria; reported, not tuned. NaN rates (no data) fail."""
    k = n_goals if k is None else k
    chance = 1 / k
    ok = lambda value, threshold, above=True: (not math.isnan(value)) and (value >= threshold if above else value <= threshold)
    explicit = {g: ok(summary["explicit"][g]["rate"], CRITERIA["explicit_min"]) for g in range(n_goals)}
    swap = {g: ok(summary["swap"]["by_instructed_goal"][g]["follow_any"]["rate"], CRITERIA["swap_follow_min"])
            for g in range(n_goals)}
    vague = {g: ok(summary["vague"]["pooled"]["achieved_any"][g]["rate"], chance, above=False) for g in range(n_goals)}
    gap = {g: ok(summary["gap"][g], CRITERIA["gap_min"]) for g in range(n_goals)}
    return dict(
        thresholds=dict(CRITERIA, vague_max=chance, k=k),
        explicit_per_goal=explicit, explicit_pass=all(explicit.values()),
        swap_overall_pass=ok(summary["swap"]["overall_follow_any"]["rate"], CRITERIA["swap_follow_min"]),
        swap_per_goal=swap, vague_per_goal=vague, vague_pass=all(vague.values()),
        gap_per_goal=gap, gap_pass=all(gap.values()),
        goals_passing_all=[g for g in range(n_goals) if explicit[g] and swap[g] and vague[g] and gap[g]])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
