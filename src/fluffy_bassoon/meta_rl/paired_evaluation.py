"""One case/noise/attempt schedule and readout for frozen-motor gate comparisons.

Scene reset/pairing and rollout are supplied by the qualified simulator adapter.
All arms consume the same schedule. Draws/attempts stay clustered by case in
inference; a best-of-N simulator oracle is never an executable selector.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random


CONTROLS = ("full", "memory_reset", "reward_blind")


def _seed(*parts):
    data = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class EvaluationPlan:
    tasks: tuple[str, ...]
    cases: int = 200
    draws: int = 1
    attempts: int = 5
    horizon: int = 600
    chunk_size: int = 50
    seed: int = 20261102
    native_seed_start: int = 810000

    def __post_init__(self):
        if (not self.tasks or len(set(self.tasks)) != len(self.tasks)
                or any(not isinstance(t, str) or not t for t in self.tasks)
                or any(type(x) is not int or x < 1 for x in
                       (self.cases, self.draws, self.attempts, self.horizon, self.chunk_size))
                or self.horizon % self.chunk_size or self.cases >= 10000
                or type(self.seed) is not int or type(self.native_seed_start) is not int
                or self.native_seed_start < 0):
            raise ValueError("Require distinct tasks and complete positive evaluation dimensions")

    def schedule(self):
        return [dict(task=task, case_id=f"{task}:{case}", draw=draw,
                     case=dict(task=task, bank="paired_evaluation", split="target",
                               native_seed=self.native_seed_start + 10000 * ti + case,
                               reset_ordinal=0, slot=case, counted=True,
                               layout_source="native_target_constructor_seed_first_reset"),
                     attempts=[dict(attempt=attempt + 1,
                                    motor_seeds=[_seed(self.seed, task, case, draw, attempt, chunk)
                                                 for chunk in range(self.horizon // self.chunk_size)])
                               for attempt in range(self.attempts)])
                for ti, task in enumerate(self.tasks) for case in range(self.cases)
                for draw in range(self.draws)]

    def manifest(self):
        schedule = self.schedule()
        return dict(format="frozen_motor_paired_evaluation_v1", config=asdict(self),
                    schedule_sha256=hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
                    controls=list(CONTROLS), schedule=schedule)


def evaluate(plan, controller_factory, rollout, *, modes=CONTROLS, episode_callback=None):
    """Run actual attempts; every arm resets physically on the canonical case.

    controller_factory(mode) returns an actor with begin_trial/begin_attempt.
    rollout(actor, case, motor_seeds) runs the full fixed horizon, validates
    canonical state/cameras, installs feedback after execution, and returns
    simulator_success_once/final_success/first_success_step/pairing_sha256.
    These diagnostics must remain outside actor observations and reward input.
    """
    if not modes or len(set(modes)) != len(modes) or any(m not in CONTROLS for m in modes):
        raise ValueError("Unknown or repeated intervention")
    rows, canonical = [], {}
    for item in plan.schedule():
        pairing = canonical.get(item["case_id"])
        for mode in modes:
            actor = controller_factory(mode)
            actor.begin_trial()
            for attempt in item["attempts"]:
                actor.begin_attempt()
                result = rollout(actor, dict(item["case"]), list(attempt["motor_seeds"]))
                if (not result.get("completed") or result.get("controls") != plan.horizon
                        or not result.get("pairing_sha256")
                        or type(result.get("simulator_success_once")) is not bool
                        or type(result.get("simulator_final_success")) is not bool):
                    raise ValueError("Incomplete, unpaired or success-truncated evaluation episode")
                pairing = result["pairing_sha256"] if pairing is None else pairing
                canonical[item["case_id"]] = pairing
                if result["pairing_sha256"] != pairing:
                    raise ValueError("Physical canonical case changed between draws/attempts/controls")
                first = result.get("first_success_step")
                if ((first is not None) != result["simulator_success_once"]
                        or first is not None and (type(first) is not int or not 1 <= first <= plan.horizon)):
                    raise ValueError("Native first-success trace does not match outcome")
                row = dict(task=item["task"], case_id=item["case_id"], draw=item["draw"],
                           mode=mode, attempt=attempt["attempt"], **result)
                rows.append(row)
                if episode_callback is not None:
                    episode_callback(row)
    return rows


def paired_readout(rows, left, right, *, attempt, metric="simulator_success_once", seed=0, bootstraps=2000):
    """Case-cluster bootstrap CI and exact two-sided case sign-randomization.

    Call separately per task. With one draw this reduces to exact McNemar.
    Multiple draws do not increase the number of independent cases.
    """
    if left == right or type(bootstraps) is not int or bootstraps < 1:
        raise ValueError("Require distinct arms and positive bootstrap count")
    selected = [r for r in rows if r["mode"] in (left, right) and r["attempt"] == attempt]
    if len({r["task"] for r in selected}) != 1:
        raise ValueError("Read out one nonempty task at a time")
    banks = {left: {}, right: {}}
    for row in selected:
        key = (row["case_id"], row["draw"])
        if key in banks[row["mode"]] or type(row[metric]) is not bool:
            raise ValueError("Duplicate or invalid endpoint")
        banks[row["mode"]][key] = int(row[metric])
    if not banks[left] or banks[left].keys() != banks[right].keys():
        raise ValueError("Require every paired case/draw in both arms")
    groups = {}
    for key, value in banks[left].items():
        groups.setdefault(key[0], []).append(value - banks[right][key])
    counts = {len(x) for x in groups.values()}
    if len(counts) != 1:
        raise ValueError("Every case must have the same complete draw count")
    draws = counts.pop()
    sums = [sum(x) for x in groups.values()]
    differences = [x / draws for x in sums]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                   for _ in range(bootstraps))
    distribution = {0: 1}
    for value in sums:
        next_distribution = {}
        for total, n in distribution.items():
            for direction in (-1, 1):
                key = total + direction * value
                next_distribution[key] = next_distribution.get(key, 0) + n
        distribution = next_distribution
    observed = abs(sum(sums))
    p = sum(n for x, n in distribution.items() if abs(x) >= observed) / 2**len(sums)
    return dict(cases=len(groups), draws_per_case=draws, attempt=attempt, metric=metric,
                left_rate=sum(banks[left].values()) / len(banks[left]),
                right_rate=sum(banks[right].values()) / len(banks[right]),
                difference=sum(differences) / len(differences),
                case_bootstrap_ci95=[means[int(.025 * bootstraps)], means[min(int(.975 * bootstraps), bootstraps - 1)]],
                exact_case_sign_randomization_p=p, multiplicity="unadjusted; declare primary contrasts before collection")
