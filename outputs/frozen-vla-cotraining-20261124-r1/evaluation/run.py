"""Run the complete locked native panel or its cue-free CPU interface fixture."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

from check_fit import inspect, sha
from panel import MODES, cases, key, require, save
from rollout import episode
from ur5e_sim_worker import UR5eSimulator
from verify import validate, summarize


def main(args):
    here = Path(__file__).resolve().parent
    pin = subprocess.check_output(["git", "-C", str(here), "rev-parse", "HEAD"], text=True).strip()
    require(pin == args.pin, "checkout revision")
    fixture = args.fixture
    metadata = dict(pin=pin, registered_panel_sha256=sha(here/"PANEL.md"), host=socket.gethostname(),
                    fixture=fixture, cpu_affinity=sorted(os.sched_getaffinity(0)))
    if fixture:
        from fixture import Fixture
        require(len(os.sched_getaffinity(0)) <= 2 and os.environ["CUDA_VISIBLE_DEVICES"] == "" and
                os.environ["JAX_PLATFORMS"] == "cpu", "two-core CPU-only fixture")
        motor = Fixture()
    else:
        from cotrain_policy import LockedPolicy
        import jax
        require(len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu", "allocated CUDA device")
        require(args.fit and args.fit_sha and sha(args.fit/"fit.json") == args.fit_sha, "source checkpoint lock")
        metadata["offline_fit"] = inspect(args.fit, args.fit_sha, args.data_manifest)
        from deployed import Deployed
        motor = Deployed(LockedPolicy.load(args.fit))
        metadata.update(fit=str(args.fit.resolve()), fit_sha256=args.fit_sha)
    save(args.out/"run.json", metadata)
    before = motor.digests()
    if not fixture:
        require(before["fit_file"] == args.fit_sha and before["resident_stock"] and before["resident_heads"]
                and before["normalization"], "live parameter and normalization hashes")
    save(args.out/"frozen-before.json", before)
    (args.out/"episodes").mkdir()
    # The render child gets the allocation's validated EGL index; inference retains its CUDA visibility.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        simulator = UR5eSimulator(allow_mesa=fixture)
    with simulator as sim:
        if fixture:
            from fixture import native_timeout
            save(args.out/"native-timeout-fixture.json", native_timeout(sim))
        else:
            from ur5e_motor import NOISE_SEED_OFFSET, to_openpi_observation
            import numpy as np
            historical = dict(arm="Panda", suite="libero_goal", task_id=3, variant="BASE", init_state_id=14, seed=41810314)
            _, opening = sim.call("create", dict(historical, embodiment="nom"))
            item = to_openpi_observation(opening["observation"], opening["language_instruction"])
            noise = np.random.default_rng(historical["seed"] + NOISE_SEED_OFFSET).standard_normal((10,32)).astype(np.float32)
            parity = motor.parity(item, noise=noise)
            save(args.out/"historical-noop.json", dict(case=historical, parity=parity))
            require(parity["deployed_noop_sha256"] == parity["repeated_stock_sha256"] == parity["stock_sha256"] and
                    parity["deployed_noop_max_abs"] == 0,
                    "deployed history graph with a zero update diverged on historical stock readback; see historical-noop.json")
        for case in cases():
            reference = None
            # Warm the release's first-time model registration before capturing the reference physics.
            sim.call("create", dict(case, embodiment="nom"))
            for mode in MODES:
                body = episode(sim, motor, case, mode, reference, fixture)
                validate(body, fixture)
                if mode == "original":
                    reference = body
                save(args.out/"episodes"/f"{key(case)}-{mode}.json", body)
                print(key(case), mode, body["native"]["controls"], flush=True)
    after = motor.digests()
    save(args.out/"frozen-after.json", after)
    require(before == after, "parameter or artifact drift")
    save(args.out/("cpu-summary.json" if fixture else "native-summary.json"), summarize(args.out, fixture))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("pin")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fit-sha")
    parser.add_argument("--data-manifest", type=Path)
    main(parser.parse_args())
