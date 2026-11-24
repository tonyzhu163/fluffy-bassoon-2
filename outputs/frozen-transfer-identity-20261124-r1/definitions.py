"""Extract BASE instructions, instance names and predicates from the pinned release only."""
import hashlib
import json
import re
from pathlib import Path

ASSETS = Path("/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69/assets")
# Authored from public tasks. Validate the carrier set against both the goal and the object roster.
PICKUPS = {
    "libero_10/5": ["black_book_1"],
    "libero_10/8": ["moka_pot_1", "moka_pot_2"],
    "libero_goal/3": ["akita_black_bowl_1"],
    "libero_goal/6": ["cream_cheese_1"],
    "libero_object/1": ["cream_cheese_1"],
    "libero_object/8": ["chocolate_pudding_1"],
    "libero_spatial/0": ["akita_black_bowl_1"],
    "libero_spatial/8": ["akita_black_bowl_1"],
}


def field(bddl, name):
    i = bddl.index("(:" + name + " ")
    start = i + 2 + len(name)
    depth = 1
    for j in range(start, len(bddl)):
        depth += (bddl[j] == "(") - (bddl[j] == ")")
        if not depth:
            return bddl[start:j].strip()
    raise ValueError(name)


def typed_names(section):
    names, pending = {}, []
    tokens = iter(section.split())
    for token in tokens:
        if token == "-":
            kind = next(tokens)
            names.update(dict.fromkeys(pending, kind))
            pending = []
        else:
            pending.append(token)
    assert not pending
    return names


def main():
    manifest = ASSETS / "task_manifest.json"
    rules = {}
    for item in json.loads(manifest.read_bytes())["tasks"]:
        task = f"{item['suite']}/{item['base_task_id']}"
        if task not in PICKUPS or item["case_id"] != "BASE":
            continue
        path = ASSETS / item["bddl_path"]
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == item["bddl_sha256"]
        data = " ".join(blob.decode().split())
        objects = typed_names(field(data, "objects"))
        fixtures = typed_names(field(data, "fixtures"))
        goal = field(data, "goal")
        placed = re.findall(r"\((?:On|In) (\w+) (\w+)\)", goal, re.I)
        assert {p for p, _ in placed} == set(PICKUPS[task]) <= objects.keys()
        destinations = {d for _, r in placed for d in objects | fixtures if r == d or r.startswith(d + "_")}
        assert len(destinations) == 1
        rules[task] = dict(prompt=field(data, "language"), goal=goal, init=field(data, "init"),
                           pickups=PICKUPS[task], receivers=sorted(destinations), objects=objects, fixtures=fixtures,
                           path=item["bddl_path"], bddl_sha256=item["bddl_sha256"])
    assert rules.keys() == PICKUPS.keys()
    print(json.dumps(dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                          release="e5500c690c1498eca1b7050350654964190cdf70", tasks=rules), indent=2))


if __name__ == "__main__":
    main()
