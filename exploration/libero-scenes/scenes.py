"""Procedural LIBERO tabletop scenes for hidden-rule sessions.

A scene is one draw on each axis in `Axes`: how many bowls, which bowl kinds,
which receptacles, which fixtures, and where everything sits. Each axis is a
tuple of options; extending the scene family means appending to one. The
output is one LIBERO task file per scene in the release's own format, plus a
manifest listing every (bowl, target) behaviour and its native predicate.

    python scenes.py --preset positions --n 64 --seed 0 --out <dir>

This file needs only numpy and runs anywhere. Nothing here has been through a
simulator: footprints and keep-out distances are approximations, and
`admit.py` is the check that a scene is physically valid.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Approximate footprint radii in metres, used only to space objects apart.
BOWLS = {"akita_black_bowl": 0.055, "white_bowl": 0.055}
RECEPTACLES = {
    "plate": 0.075,
    "glazed_rim_porcelain_ramekin": 0.045,
    "cookies": 0.06,
    "basket": 0.10,  # not a target in the release
    "chefmate_8_frypan": 0.12,  # not a target in the release
}
MARGIN = 0.03

# Fixture poses are the release's. `target` is the surface a bowl can be placed on.
FIXTURES = {
    "wooden_cabinet": dict(region="cabinet_region", xy=(0.03, -0.27), keep_out=0.22, target="top_side",
                           yaw=(2.6613777765410678, 2.7242096296128633),
                           surfaces=("top_region", "middle_region", "bottom_region", "top_side")),
    "flat_stove": dict(region="stove_region", xy=(-0.41, -0.14), keep_out=0.28, target="cook_region",
                       yaw=None, surfaces=("cook_region",)),
}

# Table-frame rectangle that contains every object centre in the release.
AREA = (-0.24, -0.08, 0.20, 0.34)
INIT_HALF = 0.01  # the release places each object in a 2 cm box
GOAL_HALF = 0.04  # and scores table targets with an 8 cm box
SIDES = {"xm": (-1, 0), "xp": (1, 0), "ym": (0, -1), "yp": (0, 1)}


@dataclass(frozen=True)
class Axes:
    bowl_counts: tuple[int, ...]
    bowl_palettes: tuple[tuple[str, ...], ...]  # each bowl's kind is drawn from the chosen palette
    receptacle_sets: tuple[tuple[str, ...], ...]
    fixture_sets: tuple[tuple[str, ...], ...]
    beside: bool  # add "beside the receptacle" table targets


_RELEASE = ("plate", "glazed_rim_porcelain_ramekin", "cookies")
_BOTH = ("wooden_cabinet", "flat_stove")
PRESETS = {
    # The release's objects with positions sampled: the smallest step away from the fixed layouts.
    "positions": Axes((2,), (("akita_black_bowl",),), (_RELEASE,), (_BOTH,), False),
    # Adds bowl count and colour, receptacle subsets, fixture subsets and beside-targets.
    "objects": Axes((2, 3), (("akita_black_bowl",), ("akita_black_bowl", "white_bowl")),
                    tuple(c for k in (2, 3) for c in itertools.combinations(_RELEASE, k)),
                    (_BOTH, ("wooden_cabinet",), ("flat_stove",), ()), True),
    # Adds two receptacle kinds the release never uses as targets.
    "extended": Axes((2, 3), (("akita_black_bowl",), ("akita_black_bowl", "white_bowl")),
                     tuple(c for k in (3, 4) for c in itertools.combinations(RECEPTACLES, k)),
                     (_BOTH, ("wooden_cabinet",), ("flat_stove",), ()), True),
}


def _instances(kinds: list[str]) -> list[tuple[str, str]]:
    """(instance, kind) with LIBERO's numbering: plate_1, akita_black_bowl_2."""
    seen: dict[str, int] = {}
    out = []
    for kind in kinds:
        seen[kind] = seen.get(kind, 0) + 1
        out.append((f"{kind}_{seen[kind]}", kind))
    return out


def _clear(xy, radius, placed, fixtures) -> bool:
    if not (AREA[0] <= xy[0] <= AREA[2] and AREA[1] <= xy[1] <= AREA[3]):
        return False
    if any(np.hypot(*np.subtract(xy, FIXTURES[f]["xy"])) < FIXTURES[f]["keep_out"] for f in fixtures):
        return False
    return all(np.hypot(*np.subtract(xy, p["xy"])) >= radius + p["radius"] + MARGIN for p in placed)


def sample_scene(axes: Axes, seed: int) -> dict:
    """One scene: the axis draws, object positions in the table frame, targets and behaviours."""
    rng = np.random.default_rng(seed)
    pick = lambda options: options[rng.integers(len(options))]
    palette = pick(axes.bowl_palettes)
    bowl_kinds = sorted(str(rng.choice(palette)) for _ in range(pick(axes.bowl_counts)))
    receptacle_kinds, fixtures = list(pick(axes.receptacle_sets)), list(pick(axes.fixture_sets))

    for _ in range(1000):  # restart the whole layout when a late object cannot be placed
        placed = []
        for role, (name, kind) in [("receptacle", x) for x in _instances(receptacle_kinds)] + \
                                  [("bowl", x) for x in _instances(bowl_kinds)]:
            radius = (RECEPTACLES if role == "receptacle" else BOWLS)[kind]
            for _ in range(200):
                xy = rng.uniform(AREA[:2], AREA[2:])
                if _clear(xy, radius, placed, fixtures):
                    placed.append(dict(name=name, kind=kind, role=role, radius=radius,
                                       xy=[round(float(v), 4) for v in xy]))
                    break
            else:
                break
        else:
            break
    else:
        raise RuntimeError(f"no collision-free layout for seed {seed}")

    # Table targets beside each receptacle, kept only where a bowl would fit.
    bowl_r = max(BOWLS.values())
    beside = []
    if axes.beside:
        for p in (p for p in placed if p["role"] == "receptacle"):
            for side, (dx, dy) in SIDES.items():
                reach = p["radius"] + bowl_r + 0.04
                xy = [round(p["xy"][0] + dx * reach, 4), round(p["xy"][1] + dy * reach, 4)]
                # Clear of every object, bowls included, so no behaviour starts out satisfied.
                if _clear(xy, bowl_r, [dict(q, radius=q["radius"] - MARGIN) for q in placed], fixtures):
                    beside.append(dict(region=f"beside_{p['name']}_{side}", of=p["name"], side=side, xy=xy))

    targets = [dict(name=p["name"], kind="receptacle") for p in placed if p["role"] == "receptacle"]
    targets += [dict(name=f"{f}_1_{FIXTURES[f]['target']}", kind="fixture") for f in fixtures]
    targets += [dict(name=f"main_table_{b['region']}", kind="beside", of=b["of"], side=b["side"]) for b in beside]
    bowls = [p["name"] for p in placed if p["role"] == "bowl"]
    behaviours = [dict(bowl=b, target=t["name"], predicate=["on", b, t["name"]]) for b in bowls for t in targets]
    return dict(seed=seed, fixtures=fixtures, objects=placed, beside=beside, targets=targets, behaviours=behaviours)


def _region(name: str, target: str, box=None, yaw=None) -> str:
    text = f"      ({name}\n          (:target {target})\n"
    if box is not None:
        text += f"          (:ranges (\n              ({' '.join(repr(float(v)) for v in box)})\n            )\n          )\n"
    if yaw is not None:
        text += f"          (:yaw_rotation (\n              ({yaw[0]!r} {yaw[1]!r})\n            )\n          )\n"
    return text + "      )\n"


def _box(xy, half) -> list[float]:
    return [round(xy[0] - half, 4), round(xy[1] - half, 4), round(xy[0] + half, 4), round(xy[1] + half, 4)]


def to_bddl(scene: dict) -> str:
    """The scene as a LIBERO task file, laid out like the Behavior Prompting release's.

    The goal line is a placeholder so the file loads. Outcomes come from
    evaluating every behaviour's predicate on the realized state.
    """
    regions = ""
    for p in scene["objects"]:
        regions += _region(f"{p['name']}_init_region", "main_table", _box(p["xy"], INIT_HALF))
    for f in scene["fixtures"]:
        regions += _region(FIXTURES[f]["region"], "main_table", _box(FIXTURES[f]["xy"], INIT_HALF), FIXTURES[f]["yaw"])
    for f in scene["fixtures"]:
        for surface in FIXTURES[f]["surfaces"]:
            regions += _region(surface, f"{f}_1")
    for b in scene["beside"]:
        regions += _region(b["region"], "main_table", _box(b["xy"], GOAL_HALF))

    by_kind: dict[str, list[str]] = {}
    for p in scene["objects"]:
        by_kind.setdefault(p["kind"], []).append(p["name"])
    objects = "".join(f"    {' '.join(names)} - {kind}\n" for kind, names in sorted(by_kind.items()))
    fixtures = "    main_table - table\n" + "".join(f"    {f}_1 - {f}\n" for f in scene["fixtures"])
    init = "".join(f"    (On {p['name']} main_table_{p['name']}_init_region)\n" for p in scene["objects"])
    init += "".join(f"    (On {f}_1 main_table_{FIXTURES[f]['region']})\n" for f in scene["fixtures"])
    first = scene["behaviours"][0]
    return (
        "(define (problem LIBERO_Tabletop_Manipulation)\n"
        "  (:domain robosuite)\n"
        "  (:language Do the demonstrated task)\n"
        f"    (:regions\n{regions}    )\n\n"
        f"  (:fixtures\n{fixtures}  )\n\n"
        f"  (:objects\n{objects}  )\n\n"
        f"  (:obj_of_interest\n    {first['bowl']}\n    {first['target']}\n  )\n\n"
        f"  (:init\n{init}  )\n\n"
        f"  (:goal\n    (And (On {first['bowl']} {first['target']}))\n  )\n\n"
        ")\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=PRESETS, default="positions")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="scene i uses seed + i")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    scenes = []
    for i in range(args.n):
        scene = dict(index=i, **sample_scene(PRESETS[args.preset], args.seed + i))
        text = to_bddl(scene)
        scene["bddl"] = f"scene-{i:04d}.bddl"
        scene["bddl_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        (args.out / scene["bddl"]).write_text(text)
        scenes.append(scene)
    manifest = dict(preset=args.preset, axes=asdict(PRESETS[args.preset]), seed=args.seed, frame="main_table",
                    area=AREA, scenes=scenes)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    count = lambda key: sorted({len(s[key]) for s in scenes})
    print(f"{args.n} scenes, preset {args.preset}: behaviours per scene {count('behaviours')}, "
          f"objects {count('objects')}, fixtures {count('fixtures')}")


if __name__ == "__main__":
    main()
