"""Reduce the predeclared matched 3-task ID panel from archived per-bank results."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tarfile

TASKS = ("OpenDrawer", "CloseToasterOvenDoor", "CoffeeSetupMug")


def reduce(read_bytes, method):
    root = f"readouts/kv_bridge/{method}/ID"
    binding_bytes = read_bytes(root + "/BINDING.json")
    binding = json.loads(binding_bytes)
    assert (binding["interface"], binding["objective"], binding["split"], binding["cases_per_task"],
            binding["deployment_optimizer_updates"]) == ("kv_bridge", method, "ID", 50, 0)
    binding_sha = hashlib.sha256(binding_bytes).hexdigest()
    output, contrasts = {}, []
    for task_index, task in enumerate(TASKS):
        arms = {}
        for mode in ("full", "episode"):
            outcomes = {}
            for start in range(0, 50, 8):
                wave = json.loads(read_bytes(f"{root}/{mode}/{task}/bank-{start:03d}/RESULT.json"))
                assert (wave["binding_sha256"], wave["task"], wave["mode"], wave["method"],
                        wave["native_parameters_frozen"], wave["optimizer_updates"]) == (binding_sha, task, mode, method, True, 0)
                assert wave["denominators"] == min(8, 50-start)
                for ep in wave["episodes"]:
                    c, n = ep["case"], ep["attempt"]
                    slot = c["slot"]
                    assert c["task"] == task and start <= slot < min(50,start+8) and ep["completed"]
                    assert c["native_seed"] == 64105000 + 10000*task_index + slot and ep["horizon"] == 600
                    assert 1 <= n <= 6 and type(ep["success_once"]) is bool and (slot,n) not in outcomes
                    outcomes[slot,n] = int(ep["success_once"])
            assert len(outcomes) == 300
            arms[mode] = outcomes
        f, r = arms["full"], arms["episode"]
        deltas = [f[s,6] - r[s,6] for s in range(50)]
        contrasts.append(deltas)
        output[task] = dict(paired_cases=50,
            query_successes={k:[sum(v[s,q] for s in range(50)) for q in range(1,7)] for k,v in arms.items()},
            sixth_episode_wins=sum(x==1 for x in deltas), sixth_episode_losses=sum(x==-1 for x in deltas),
            sixth_episode_delta_pp=2*sum(deltas))
    # Seed/order are identical on every method so cross-method comparisons can reuse the draws.
    rng = random.Random(20261109)
    samples = sorted(100/150*sum(sum(rng.choices(v,k=50)) for v in contrasts) for _ in range(10000))
    return dict(method=method, selected_sha256=binding["selected_candidate_sha256"], tasks=output,
                ID_mean_sixth_episode_delta_pp=100/150*sum(map(sum,contrasts)),
                ID_mean_percentile95_pp=[samples[249],samples[9749]], bootstrap_seed=20261109,
                bootstrap_replicates=10000, uncertainty_scope="case sampling, conditional on this fixed fit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path, help="Root directory or metadata tar containing readouts/")
    parser.add_argument("method", choices=("ad","dpt","dit"))
    args = parser.parse_args()
    if args.panel.is_dir():
        result = reduce(lambda n:(args.panel/n).read_bytes(), args.method)
    else:
        with tarfile.open(args.panel) as tar:
            result = reduce(lambda n:tar.extractfile(n).read(), args.method)
    print(json.dumps(result, indent=2))
