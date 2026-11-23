"""Check the complete CSV/trace transport and compare its summary to the released CLI on a fixture."""
import argparse
import ast
import contextlib
import csv
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import reference as ref


def main(out):
    with tempfile.TemporaryDirectory(dir=ref.ROOT, prefix="reference-metrics-cpu-") as tmp:
        ref.STORE, ref.NATIVE = Path(tmp), Path(tmp) / "runs"
        ref.configure()
        from reground.envs.libero_env import max_steps, task_suite
        from reground.eval import jobs
        from reground.utils import cfg_id, noise_seed
        for method in ("base", "reground"):
            for j in ref.units(method, ref.SEEDS):
                d = ref.output(j)
                d.mkdir(parents=True)
                traces = {}
                with (d / "results.csv").open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=jobs.FIELDS)
                    writer.writeheader()
                    for task, init, rep in j["episodes"]:
                        success = int((init + j["seed"]) % 5 < (1 if method == "base" else 3))
                        steps = 1 if success else max_steps(task)
                        contact = int((init + j["seed"]) % 3 == 0)
                        row = dict.fromkeys(jobs.FIELDS, 0)
                        row.update(policy="pi05", task=task, suite=task_suite(task), shift="rot_z", level=90,
                                   method=method, cfg_id=cfg_id(j["cfg"]), adapt_seed=j["seed"], split="test",
                                   init_id=init, rep=rep, noise_seed=noise_seed(init, rep), success=success,
                                   steps=steps, contact_steps=contact, unsafe=contact, alarm_step=-1)
                        writer.writerow(row)
                        for field, shape in (("X", (steps + 1, 25)), ("A", (steps, 7)), ("A_vla", (steps, 7)),
                                             ("ref", (steps, 3)), ("speed", (steps,))):
                            traces[f"{task}__{init}__{rep}__{field}"] = np.zeros(shape, dtype=np.float32)
                (d / "traces").mkdir()
                np.savez_compressed(d / "traces/batch.npz", **traces)
                ref.write(d / "manifest.json", dict(git_sha=ref.PIN, vla_revision=ref.POLICY,
                                                    cfg=j["cfg"], n_grid=len(j["full_episodes"])))
        with contextlib.redirect_stdout(io.StringIO()):
            ref.summarize()
        table = ref.read(ref.STORE / "summary.json")
        for method in table:
            stream = io.StringIO()
            argv = ["main.py", "summarize", "--cell", "rot_z/90", "--method", method]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stream):
                runpy.run_path(str(ref.SOURCE / "main.py"), run_name="__main__")
            expected = ast.literal_eval(stream.getvalue().strip().split(" ", 1)[1])
            assert {k: v for k, v in table[method].items() if k != "per_task"} == expected
            assert expected["n_eps"] == 1400
        # A corrupted unsafe flag must fail before it can enter the score.
        job = ref.units("base", [0])[0]
        d = ref.output(job)
        path = d / "results.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))
        rows[0]["unsafe"] = 1 - int(rows[0]["unsafe"])
        with path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=jobs.FIELDS)
            w.writeheader()
            w.writerows(rows)
        try:
            ref.check_eval(d, job)
        except ValueError:
            pass
        else:
            raise AssertionError("A corrupted unsafe flag escaped validation")

        attempts = ref.STORE / "budget-fixture"
        env = dict(REFERENCE_CODE_PIN="fixture", REFERENCE_PHASE="base", REFERENCE_STARTED="0")
        with patch.dict(os.environ, env):
            def reserve(name):
                path = attempts / name
                path.mkdir(parents=True)
                ref.reserve(SimpleNamespace(attempt=path, gpus=2, wall=1500, cap=1, seeds=[0]))
            reserve("first")
            try:
                reserve("over-budget")
            except ValueError:
                pass
            else:
                raise AssertionError("Outstanding reference reservation was double-spent")
            ref.write(attempts / "first/exit.json", {"allocated_gpu_seconds": 400})
            reserve("next")
            assert ref.read(attempts / "next/reservation.json")["prior_gpu_seconds"] == 400
        ref.write(out, dict(fixture=True, upstream=ref.PIN, per_method_release_cli_exact=True,
                            corrupted_unsafe_rejected=True, budget_reservation_and_settlement=True,
                            summary=table, gpu_hours=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
