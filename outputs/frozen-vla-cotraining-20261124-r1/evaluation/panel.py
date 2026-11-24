"""Preregistered official BASE tasks, released numerically fingerprinted rows 46–49."""
import json
from pathlib import Path

MODES = ("original", "policy_only", "adapter_only", "joint", "joint_zero", "joint_lagged")
TASKS = {
    ("libero_goal", 3): (42810300, 300, "open the top layer of the drawer and put the bowl inside", (
        "f2733f57ba35218560fd079f81edfafffaf158788b244d03cb698224420c0720",
        "b74398b355e2f1222fc9a8a35a5853735c36a759a311cf395ac853fd951c29b8",
        "44f20cffbfae56bffa6c2fbcc43e936b3bfd1adb5ab35317b5e9235dddb30f60",
        "cf86c5b5222b47c570a483f0efb4d7e9aa5051ebf5df12cac397cf62f820dc81")),
    ("libero_goal", 6): (42810600, 300, "put the cream cheese on the bowl", (
        "27e9237d7195a146dced40f1b7cfeeef27a01478dd51e7028529054abf31efe0",
        "ece4cd1a23f25fd9e8db9b0da50f457217dfb641f13aec5b2ec39c06b427f700",
        "072b44316bfdf52bc03ddb0e171adb9e5f6647befff7aa87f528ea2c5d3c7761",
        "d11703ff9583e11f05e364b341a0c695529e38d237052f29d9d8d86067a39718")),
    ("libero_10", 8): (42800800, 520, "put both moka pots on the stove", (
        "28fe3abb2d39dabd37c9dbf77cd5d011a6b853db44514e4b71621bd1c9db9e09",
        "e48c76396fec5968372cd8522002209a51221321b196bb9c3db8a81e0d6a4107",
        "4a65421d6acdc67531f1ee304f418d38daf48a9a4cf0b68a3385862dc40e2195",
        "3a1fbf0f5085241faec75f67f0255b34aa3f2f704d99e7bf24ba55fbc5487ac8")),
}
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
MANIFEST = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"


def cases():
    for (suite, task), (seed_base, *_) in TASKS.items():
        for row in range(46, 50):
            for arm in (("Kinova3", "Panda") if row >= 48 else ("Kinova3",)):
                yield dict(arm=arm, suite=suite, task_id=task, variant="BASE", init_state_id=row, seed=seed_base + row)


def key(case):
    return "{arm}-{suite}-{task_id}-{init_state_id}".format(**case)


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    path = Path(path)
    require(not path.exists(), f"Refusing to overwrite {path}")
    tmp = path.with_suffix(".tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    tmp.replace(path)
