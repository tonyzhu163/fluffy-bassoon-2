"""Instruction-swap check: does pi0.5's language instruction choose the manipulated object?

Scenes come from RLinf's own `PutOnPlateInScene25MultiCarrot-v1` (e1a82eb): a named
"carrot" slot object, one extra object from the same 25-object set and one plate.
A subclass registered here (no RLinf patch) adds two things:

- an instruction override, so the exact same scene/reset/noise can be run with the
  instruction naming the slot-0 object, the slot-1 object, or neither;
- per-step outcomes for BOTH objects (grasped, on plate by RLinf's own success test,
  object-to-plate distance), since RLinf's evaluate() only tracks the slot-0 object.

Pure-Python pieces (episode ids, plans, outcome choice, metrics) have no simulator
dependency and are covered by CPU tests.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from . import maniskill_pi05 as ms

ENV_ID = "PutOnPlateInScene25InstructionSwap-v1"
SLOTS = 2
CONDITIONS = ("name_slot0", "name_slot1", "vague")
VAGUE_INSTRUCTION = "put object on plate"


def instruction(object_name, plate_name="plate"):
    """RLinf's PutOnPlateInScene25 template: f"put {carrot_name} on {plate_name}"."""
    return f"put {object_name} on {plate_name}"


# ----------------------------------------------------------------------------- scene ids

def check_multi_infos(infos):
    """RLinf MultiCarrot basic_obj_infos with obj_set="train" and one plate."""
    lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset = infos
    if (lc, lc_offset, lo, lo_offset, lp, lp_offset, le, le_offset) != (16, 0, 16, 0, 1, 0, 15, 0) or l1 < 1 or l2 != 4:
        raise ValueError(f"Unexpected MultiCarrot train-set infos {infos}")
    return lc, lo, lp, l1, l2, le


def multi_episode_id(infos, slot0, slot1, overlay, position, orientation):
    """Inverse of PutOnPlateInScene25MultiCarrot._initialize_episode_pre (obj_set=train).

    carrot = id // (le*lp*lo*l1*l2); extra = (carrot + (id // (lp*lo*l1*l2)) % le + 1) % lc.
    """
    lc, lo, lp, l1, l2, le = check_multi_infos(infos)
    if not (0 <= slot0 < lc and 0 <= slot1 < lc and slot0 != slot1 and 0 <= overlay < lo
            and 0 <= position < l1 and 0 <= orientation < l2):
        raise ValueError("MultiCarrot scene index out of range")
    extra_offset = (slot1 - slot0 - 1) % lc
    return ((((slot0*le + extra_offset)*lp + 0)*lo + overlay)*l1 + position)*l2 + orientation


def decode_multi_episode_id(infos, episode):
    """Same arithmetic as RLinf's _initialize_episode_pre, for checking plans."""
    lc, lo, lp, l1, l2, le = check_multi_infos(infos)
    slot0 = episode // (le*lp*lo*l1*l2)
    slot1 = (slot0 + (episode // (lp*lo*l1*l2)) % le + 1) % lc
    return dict(slot0=int(slot0), slot1=int(slot1), plate=int((episode // (lo*l1*l2)) % lp),
                overlay=int((episode // (l1*l2)) % lo), position=int((episode // l2) % l1),
                orientation=int(episode % l2))


def plan_multi_scenes(infos, pairs, per_pair, rng):
    """Scenes per unordered pair, balanced over which object occupies slot 0."""
    if per_pair % 2:
        raise ValueError("per_pair must be even to balance slot order")
    _, lo, _, l1, l2, _ = check_multi_infos(infos)
    scenes = []
    for a, b in pairs:
        for k in range(per_pair):
            slot0, slot1 = (a, b) if k % 2 == 0 else (b, a)
            overlay, pos, quat = int(rng.integers(lo)), int(rng.integers(l1)), int(rng.integers(l2))
            scenes.append(dict(pair=[int(a), int(b)], slot0=int(slot0), slot1=int(slot1), overlay=overlay,
                               position=pos, orientation=quat,
                               episode_id=multi_episode_id(infos, slot0, slot1, overlay, pos, quat)))
    return scenes


def interleave_blocks(scenes, block):
    """Order scenes so each block of `block` rows mixes all pairs (partial runs stay balanced)."""
    by_pair = defaultdict(list)
    for scene in scenes:
        by_pair[tuple(scene["pair"])].append(scene)
    queues = list(by_pair.values())
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    if len(ordered) % block:
        raise ValueError("Scene count must be a multiple of the batch size")
    return [ordered[i:i+block] for i in range(0, len(ordered), block)]


def condition_instructions(scenes, names, condition, plate_name="plate"):
    if condition == "name_slot0":
        return [instruction(names[s["slot0"]], plate_name) for s in scenes]
    if condition == "name_slot1":
        return [instruction(names[s["slot1"]], plate_name) for s in scenes]
    if condition == "vague":
        return [VAGUE_INSTRUCTION]*len(scenes)
    raise ValueError(condition)


# ------------------------------------------------------------------------- per-object outcomes

SLOT_FIELDS = dict(
    grasped_once="agent.is_grasping(object) at any step",
    first_grasp_step="first step grasped, -1 if never",
    on_plate_once="RLinf success test (bbox xy/z + contact force) for this object at any step",
    on_plate_end="RLinf success test for this object at step 80",
    first_on_plate_step="first step on plate, -1 if never",
    plate_distance_start_m="object-to-plate centre distance after the first step",
    plate_distance_min_m="minimum object-to-plate distance over the episode",
    plate_distance_end_m="object-to-plate distance at step 80")


class SlotOutcomes:
    """Per-env, per-slot outcomes from the subclass's swap_* info tensors ([envs, 2])."""

    def __init__(self, envs):
        shape = (envs, SLOTS)
        self.step = 0
        self.grasped = torch.zeros(shape, dtype=torch.bool)
        self.first_grasp = torch.full(shape, -1, dtype=torch.long)
        self.on_once = torch.zeros(shape, dtype=torch.bool)
        self.on_end = torch.zeros(shape, dtype=torch.bool)
        self.first_on = torch.full(shape, -1, dtype=torch.long)
        self.start = torch.full(shape, float("nan"), dtype=torch.float64)
        self.minimum = torch.full(shape, float("inf"), dtype=torch.float64)
        self.end = torch.full(shape, float("nan"), dtype=torch.float64)
        self.slot0_success_mismatch = 0

    def update(self, info):
        grasp = info["swap_grasp"].detach().cpu().bool()
        on = info["swap_on_plate"].detach().cpu().bool()
        dist = info["swap_plate_dist"].detach().cpu().to(torch.float64)
        if not (grasp.shape == on.shape == dist.shape == self.grasped.shape) or not torch.isfinite(dist).all():
            raise ValueError("Require finite [envs, 2] swap outcomes")
        self.slot0_success_mismatch += int((info["success"].detach().cpu().bool() != on[:, 0]).sum())
        self.first_grasp[grasp & (self.first_grasp < 0)] = self.step
        self.first_on[on & (self.first_on < 0)] = self.step
        self.grasped |= grasp
        self.on_once |= on
        self.on_end = on
        if self.step == 0:
            self.start = dist.clone()
        self.minimum = torch.minimum(self.minimum, dist)
        self.end = dist
        self.step += 1

    def rows(self):
        if self.step != ms.STEPS:
            raise RuntimeError(f"Episode ended after {self.step} of {ms.STEPS} steps")
        rows = []
        for i in range(self.grasped.shape[0]):
            rows.append({f"slot{s}_{key}": value for s in range(SLOTS) for key, value in (
                ("grasped_once", bool(self.grasped[i, s])), ("first_grasp_step", int(self.first_grasp[i, s])),
                ("on_plate_once", bool(self.on_once[i, s])), ("on_plate_end", bool(self.on_end[i, s])),
                ("first_on_plate_step", int(self.first_on[i, s])),
                ("plate_distance_start_m", float(self.start[i, s])),
                ("plate_distance_min_m", float(self.minimum[i, s])),
                ("plate_distance_end_m", float(self.end[i, s])))})
        return rows


def chosen_slot(row):
    """Which object the policy acted on: placed > first grasped > none (ties: earliest step)."""
    for flag, step in (("on_plate_once", "first_on_plate_step"), ("grasped_once", "first_grasp_step")):
        hits = [s for s in range(SLOTS) if row[f"slot{s}_{flag}"]]
        if hits:
            return min(hits, key=lambda s: (row[f"slot{s}_{step}"], s))
    return None


def chosen_object(row):
    slot = chosen_slot(row)
    return None if slot is None else row[f"slot{slot}"]


def placed_objects(row):
    return [row[f"slot{s}"] for s in range(SLOTS) if row[f"slot{s}_on_plate_once"]]


def named_slot(condition):
    return {"name_slot0": 0, "name_slot1": 1}.get(condition)


# ---------------------------------------------------------------------------------- metrics

def _rate(values):
    values = list(values)
    return dict(rate=float(np.mean(values)) if values else None, n=len(values), k=int(sum(values)))


def summarize(single_rows, multi_rows, names):
    """Metrics of the instruction-swap protocol. Rows are the saved per-episode records."""
    per_object = defaultdict(dict)
    for obj in sorted({r["object"] for r in single_rows}):
        per_object[obj]["single_success"] = _rate(r["success_once"] for r in single_rows if r["object"] == obj)
    by_scene = defaultdict(dict)
    for row in multi_rows:
        by_scene[row["scene_key"]][row["condition"]] = row
    complete = {k: v for k, v in by_scene.items() if set(v) == set(CONDITIONS)}
    explicit = defaultdict(list)
    explicit_clean = defaultdict(list)
    explicit_wrong = defaultdict(list)
    vague_placed = defaultdict(list)
    vague_choice = Counter()
    vague_choice_by_pair = defaultdict(Counter)
    pairs = []
    for key, conditions in sorted(complete.items()):
        for condition in ("name_slot0", "name_slot1"):
            row = conditions[condition]
            slot = named_slot(condition)
            target = row[f"slot{slot}"]
            explicit[target].append(row[f"slot{slot}_on_plate_once"])
            explicit_clean[target].append(row[f"slot{slot}_on_plate_once"] and not row[f"slot{1-slot}_on_plate_once"])
            explicit_wrong[target].append(row[f"slot{1-slot}_on_plate_once"])
        vague = conditions["vague"]
        for s in range(SLOTS):
            vague_placed[vague[f"slot{s}"]].append(vague[f"slot{s}_on_plate_once"])
        choice = chosen_object(vague)
        label = "none" if choice is None else names[choice]
        vague_choice[label] += 1
        vague_choice_by_pair["/".join(names[o] for o in vague["pair"])][label] += 1
        a, b = conditions["name_slot0"], conditions["name_slot1"]
        chosen_a, chosen_b = chosen_object(a), chosen_object(b)
        pairs.append(dict(scene_key=key, pair=a["pair"], slot0=a["slot0"], slot1=a["slot1"],
                          chosen_when_slot0_named=chosen_a, chosen_when_slot1_named=chosen_b,
                          chosen_vague=choice,
                          changes=chosen_a is not None and chosen_b is not None and chosen_a != chosen_b,
                          follows=chosen_a == a["slot0"] and chosen_b == a["slot1"],
                          placed_follows=(placed_objects(a) == [a["slot0"]] and placed_objects(b) == [a["slot1"]]),
                          any_grasp=any(r[f"slot{s}_grasped_once"] for r in (a, b) for s in range(SLOTS))))
    for obj in sorted(set(explicit) | set(per_object)):
        record = per_object[obj]
        record["name"] = names[obj]
        record["explicit_success"] = _rate(explicit[obj])
        record["explicit_clean_success"] = _rate(explicit_clean[obj])
        record["explicit_other_object_placed"] = _rate(explicit_wrong[obj])
        record["vague_success"] = _rate(vague_placed[obj])
        e, v = record["explicit_success"]["rate"], record["vague_success"]["rate"]
        record["gap_explicit_minus_vague"] = None if e is None or v is None else e - v
    grasped_pairs = [p for p in pairs if p["any_grasp"]]
    swap = dict(
        pairs=len(pairs), pairs_with_any_grasp=len(grasped_pairs),
        changes_all=_rate(p["changes"] for p in pairs), changes_given_grasp=_rate(p["changes"] for p in grasped_pairs),
        follows_all=_rate(p["follows"] for p in pairs), follows_given_grasp=_rate(p["follows"] for p in grasped_pairs),
        placed_follows_all=_rate(p["placed_follows"] for p in pairs))
    acted = sum(c for label, c in vague_choice.items() if label != "none")
    pair_majority = {}
    for pair, counts in vague_choice_by_pair.items():
        pair_acted = sum(c for label, c in counts.items() if label != "none")
        pair_majority[pair] = (max(c for label, c in counts.items() if label != "none")/pair_acted
                               if pair_acted else None)
    majority_hits = sum(max((c for label, c in counts.items() if label != "none"), default=0)
                        for counts in vague_choice_by_pair.values())
    vague = dict(choice_counts=dict(vague_choice), choice_by_pair={k: dict(v) for k, v in vague_choice_by_pair.items()},
                 scenes=len(complete), acted=acted,
                 majority_share_by_pair=pair_majority,
                 majority_share_among_acted=majority_hits/acted if acted else None,
                 slot0_share_among_acted=(sum(1 for p in pairs if p["chosen_vague"] is not None and p["chosen_vague"] == p["slot0"])/acted
                                          if acted else None))
    criteria = criteria_report(per_object, swap, vague)
    return dict(per_object={str(k): v for k, v in sorted(per_object.items())}, swap=swap, vague=vague,
                criteria=criteria, incomplete_scenes=len(by_scene)-len(complete), pair_records=pairs)


def criteria_report(per_object, swap, vague):
    """Provisional pass criteria fixed before the run; reported, never tuned."""
    used = [o for o, r in per_object.items() if r.get("explicit_success", {}).get("n")]
    explicit_ok = {per_object[o]["name"]: (per_object[o]["explicit_success"]["rate"] or 0) >= .6 for o in used}
    gaps = {per_object[o]["name"]: per_object[o]["gap_explicit_minus_vague"] for o in used}
    share = vague["majority_share_among_acted"]
    return dict(
        explicit_success_ge_60pct_each=dict(passed=bool(explicit_ok) and all(explicit_ok.values()), per_object=explicit_ok),
        swap_follow_ge_80pct=dict(passed=(swap["follows_all"]["rate"] or 0) >= .8,
                                  follows_all=swap["follows_all"]["rate"], follows_given_grasp=swap["follows_given_grasp"]["rate"],
                                  changes_all=swap["changes_all"]["rate"]),
        vague_near_chance=dict(note="two objects per scene; chance = 0.5 per object among acted episodes; "
                                    "majority share = within each pair, the more-chosen object's share, pooled",
                               majority_object_share_among_acted=share,
                               majority_share_by_pair=vague["majority_share_by_pair"],
                               slot0_share_among_acted=vague["slot0_share_among_acted"]),
        gap_ge_30_points_each=dict(passed=bool(gaps) and all(g is not None and g >= .3 for g in gaps.values()), per_object=gaps))


# ------------------------------------------------------------------------- simulator pieces

def register_env():
    """Register the instruction-swap subclass of RLinf's MultiCarrot scene (idempotent)."""
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env as register
    if ENV_ID in REGISTERED_ENVS:
        return
    from mani_skill.utils.geometry import rotation_conversions
    from rlinf.envs.sim.maniskill.tasks.variants.put_on_plate_25_multi_carrot import (
        PutOnPlateInScene25MultiCarrot)

    corners = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])

    class InstructionSwapScene(PutOnPlateInScene25MultiCarrot):
        instruction_override = None

        def set_instruction_override(self, instructions):
            if instructions is not None and len(instructions) != self.num_envs:
                raise ValueError("One instruction per env")
            self.instruction_override = None if instructions is None else [str(x) for x in instructions]

        def get_language_instruction(self):
            if self.instruction_override is None:
                return super().get_language_instruction()
            return list(self.instruction_override)

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            names = [self.carrot_names[i] for i in self.select_extra_ids]
            actors = [self.objs_carrot[n] for n in names]
            q = torch.stack([a.pose.q[i] for i, a in enumerate(actors)])
            half = torch.stack([self.model_bbox_sizes[n] for n in names]) / 2
            local = half[:, None, :] * corners.to(self.device)[None]
            rotated = torch.matmul(local, rotation_conversions.quaternion_to_matrix(q).transpose(1, 2))
            self.extra_bbox_world = rotated.max(dim=1).values - rotated.min(dim=1).values

        def _slot_state(self, ids, bbox):
            b = self.num_envs
            names = [self.carrot_names[i] for i in ids]
            actors = [self.objs_carrot[n] for n in names]
            plates = [self.objs_plate[self.plate_names[i]] for i in self.select_plate_ids]
            pos = torch.stack([a.pose.p[i] for i, a in enumerate(actors)])
            plate = torch.stack([a.pose.p[i] for i, a in enumerate(plates)])
            grasped = torch.zeros((b,), dtype=torch.bool, device=self.device)
            for idx, name in enumerate(self.model_db_carrot):
                selected = ids == idx
                if bool(selected.any()):
                    grasped = torch.where(selected, self.agent.is_grasping(self.objs_carrot[name]), grasped)
            # RLinf PutOnPlateInScene25.evaluate success test, applied to this object.
            offset = pos - plate
            plate_half, obj_half = self.plate_bbox_world / 2, bbox / 2
            xy = torch.linalg.norm(offset[:, :2], dim=1) <= plate_half.max(dim=1).values + 0.01
            z = (offset[:, 2] > 0) & (offset[:, 2] - plate_half[:, 2] - obj_half[:, 2] <= 0.05)
            force = torch.zeros((b,), dtype=torch.float32, device=self.device)
            for i in range(b):
                force[i] = torch.linalg.norm(self.scene.get_pairwise_contact_forces(actors[i], plates[i])[i])
            on = xy & z & (force > 0.03)
            return grasped, on, torch.linalg.norm(pos - plate, dim=1), pos

        def evaluate(self, success_require_src_completely_on_target=True):
            out = super().evaluate(success_require_src_completely_on_target)
            g0, on0, d0, p0 = self._slot_state(self.select_carrot_ids, self.carrot_bbox_world)
            g1, on1, d1, p1 = self._slot_state(self.select_extra_ids, self.extra_bbox_world)
            out.update(swap_grasp=torch.stack([g0, g1], 1), swap_on_plate=torch.stack([on0, on1], 1),
                       swap_plate_dist=torch.stack([d0, d1], 1), swap_pos=torch.stack([p0, p1], 1))
            return out

    register(ENV_ID, max_episode_steps=ms.STEPS, asset_download_ids=["bridge_v2_real2sim"])(InstructionSwapScene)


def env_config(rlinf_root, num_envs, seed):
    cfg = ms.env_config(rlinf_root, num_envs, seed)
    cfg.init_params.id = ENV_ID
    cfg.init_params.obj_set = "train"
    return cfg


@torch.no_grad()
def run_multi_batch(model, env, scenes, instructions, *, seed, generator, native_instruction):
    """One synchronized 80-step batch on fixed scenes; `instructions` replace the prompt.

    native_instruction=True leaves RLinf's own instruction in place and checks it equals
    `instructions` (validates the template used for the swapped/vague prompts).
    """
    from rlinf.envs.action_utils import prepare_actions
    base = env.env.unwrapped
    base.set_instruction_override(None if native_instruction else instructions)
    ids = torch.tensor([s["episode_id"] for s in scenes], device=env.device)
    obs, _ = env.reset(seed=seed, options={"episode_id": ids})
    if (base.select_carrot_ids.cpu().tolist() != [s["slot0"] for s in scenes]
            or base.select_extra_ids.cpu().tolist() != [s["slot1"] for s in scenes]):
        raise AssertionError("Environment did not reset to the planned objects")
    if list(obs["task_descriptions"]) != list(instructions):
        raise AssertionError("Observed instructions differ from the planned prompt")
    outcomes, target = SlotOutcomes(len(scenes)), ms.EpisodeOutcomes(len(scenes))
    initial = None
    generator.manual_seed(seed)
    for _ in range(ms.STEPS // ms.CHUNK):
        if list(obs["task_descriptions"]) != list(instructions):
            raise AssertionError("Instruction changed during the episode")
        actions, _ = model.predict_action_batch(obs, mode="eval", rng=generator)
        chunk = prepare_actions(raw_chunk_actions=actions, env_type="maniskill", model_type="openpi",
                                num_action_chunks=ms.CHUNK, action_dim=7, policy="widowx_bridge")
        obs_list, _, _, _, infos = env.chunk_step(chunk)
        for info in infos:
            if initial is None:
                initial = info["swap_pos"].detach().cpu()
            outcomes.update(info)
            target.update(info)
        obs = obs_list[-1]
    base.set_instruction_override(None)
    rows = []
    for i, (row, native) in enumerate(zip(outcomes.rows(), target.rows())):
        row["rlinf_slot0_success_once"] = native["success_once"]
        row["initial_positions"] = initial[i].tolist()
        rows.append(row)
    return rows, outcomes.slot0_success_mismatch


def write_jsonl(path, records):
    with Path(path).open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
