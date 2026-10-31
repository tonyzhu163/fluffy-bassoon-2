"""Paired real ML10 collection: serial CPU simulator versus persistent workers."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from .metaworld_env import benchmark_pools, make_lanes
from .metaworld_policy import make_policy
from .metaworld_train import core_settings
from .metaworld_workers import make_worker_lanes
from .ppo import collect_trials
from .screen import allocation, clean_source


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--allocated-cpus", type=int, default=16)
    args = p.parse_args(argv)
    evidence, source = allocation(), clean_source(args.source_commit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pools = benchmark_pools("ml10", 0)
    result = dict(source_commit=source, allocation=evidence, num_envs=args.num_envs,
                  attempts=5, horizon=500, replay_chunk_length=100,
                  methodology="paired frozen CTM T1 weights, identical seeds/actions; no learning", modes={})
    baseline, expected_outcomes, model = None, None, None
    for name, workers in (("serial", 0), ("parallel", args.workers)):
        started = time.perf_counter()
        factory = make_worker_lanes if workers else make_lanes
        options = dict(workers=workers, allocated_cpus=args.allocated_cpus) if workers else {}
        envs = factory(pools["train"], args.num_envs, benchmark="ml10", attempts=5, horizon=500, **options)
        startup = time.perf_counter() - started
        try:
            if model is None:
                model = make_policy("persistent_ctm", dict(core_settings("persistent_ctm"), width=84), envs[0]).cuda().eval()
                with torch.no_grad():
                    for _ in range(8):
                        model.core(torch.zeros(args.num_envs, model.core.input_dim, device="cuda"))
            torch.manual_seed(2000)
            torch.cuda.synchronize()
            timings = {}
            data = collect_trials(model, envs, np.arange(3000, 3000 + args.num_envs),
                                  replay_chunk_length=100, timing=timings)
            torch.cuda.synchronize()
            outcomes = [dict(family=e.family, outcomes=e.outcomes, returns=e.native_returns) for e in envs]
            if baseline is None:
                baseline, expected_outcomes = data, outcomes
            else:
                for key, value in baseline.items():
                    if isinstance(value, dict):
                        for field, tensor in value.items():
                            torch.testing.assert_close(data[key][field], tensor, rtol=0, atol=0)
                    elif isinstance(value, torch.Tensor):
                        torch.testing.assert_close(data[key], value, rtol=0, atol=0)
                    else:
                        assert data[key] == value
                assert outcomes == expected_outcomes
            digest = hashlib.sha256()
            for key, value in sorted(data.items()):
                if isinstance(value, torch.Tensor):
                    digest.update(key.encode())
                    digest.update(value.cpu().numpy().tobytes())
            result["modes"][name] = dict(startup_seconds=startup, collection=timings,
                trajectory_sha256=digest.hexdigest(), outcomes=outcomes,
                worker_manifest=envs[0]._worker_batch.manifest if workers else None)
            write_json(args.output_dir / "progress.json", result)
        finally:
            for env in envs:
                env.close()
    result["exact_trajectory_parity"] = True
    result["collection_speedup"] = (result["modes"]["serial"]["collection"]["total_seconds"] /
                                     result["modes"]["parallel"]["collection"]["total_seconds"])
    write_json(args.output_dir / "summary.json", result)


if __name__ == "__main__":
    main()
