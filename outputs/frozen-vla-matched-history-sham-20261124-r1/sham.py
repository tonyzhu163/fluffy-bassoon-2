"""Paired terminal experiment with an explicit, shared AOT history executable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import matched
import rollout
import verify
from check_fit import inspect, sha
from cotrain_policy import LockedPolicy, NativeSampler, array_sha, transformed
from openpi.shared.array_typing import disable_typechecking
from panel import key, require, save
from ur5e_motor import array_digest
from ur5e_sim_worker import UR5eSimulator

FIT = "4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc"
CONTRACT = "063c9640096eeabdeb560b2e8dd4bddac71c3250ed4b5636922d398d562d2999"
MODES = ("real", "sham", "plain")


class Sampler:
    def __init__(self, motor):
        self.motor = motor
        self.native = motor._native
        self.state = self.native.states["joint"]
        self.rng = jax.random.key(0)
        self.kernel = array_sha(self.native.kernels["joint"])
        self.graphs = {}
        self.graph_ids = {}

    def prepare(self, item, noise, history):
        obs, _ = transformed(self.motor.stock, item)
        args = self.state, self.rng, obs
        noise = jnp.asarray(noise)[None]
        offset = self.native.offset(self.native.heads["joint"], jnp.asarray(history)[None])
        # Materialize and validate the trained value even for sham before replacing it.
        actual = np.asarray(offset)
        require(actual.shape == (1, 10, 32) and actual.dtype == np.float32 and np.isfinite(actual).all(),
                "computed offset")
        if not self.graphs:
            # AOT lowering unflattens Observation with JAX ArgInfo placeholders.
            # OpenPI's type annotations describe concrete arrays on normal calls.
            with disable_typechecking():
                self.graphs["plain"] = self.native.plain.lower(*args, noise=noise).compile()
                self.graphs["history"] = self.native.with_history.lower(*args, noise=noise, offset=offset).compile()
            for name, exe in self.graphs.items():
                self.graph_ids[name] = dict(hlo_sha256=hashlib.sha256(exe.as_text().encode()).hexdigest(),
                                            process_object=id(exe))
        return args, noise, offset

    def sample(self, inputs, mode):
        args, noise, computed = inputs
        graph = "plain" if mode == "plain" else "history"
        offset = computed if mode == "real" else jnp.zeros_like(computed)
        kwargs = dict(noise=noise)
        if graph == "history":
            kwargs["offset"] = offset
        exe = self.graphs[graph]
        result = self.native.decode(args[2], exe(*args, **kwargs))
        receipt = dict(graph=graph, executable=self.graph_ids[graph], kernel=self.kernel,
                       computed_offset=array_digest(np.asarray(computed)),
                       passed_offset=array_digest(np.asarray(offset)) if graph == "history" else None)
        return result, receipt


class Wire(rollout.Wire):
    def infer(self, item, *, noise):
        history = self.history.vector().copy()
        inputs = self.motor.prepare(item, noise, history)
        mode = self.mode if self.controls else "plain"
        result, route = self.motor.sample(inputs, mode)
        proposal = np.asarray(result["actions"], np.float32)
        obs = {k: array_digest(v) for k, v in item.items() if k != "prompt"}
        if self.controls == 10 and self.mode == "real":
            probes = {}
            for label, query in (("real", "real"), ("sham", "sham"), ("sham_replay", "sham"), ("plain", "plain")):
                unissued, receipt = self.motor.sample(inputs, query)
                a = np.asarray(unissued["actions"], np.float32)
                probes[label] = dict(route=receipt, proposal=array_digest(a), actions=a.tolist())
            self.mechanism = dict(at_control=10, obs=obs, prompt=item["prompt"], noise=array_digest(noise),
                                  rng=np.asarray(jax.random.key_data(self.motor.rng)).tolist(),
                                  history=array_digest(history), probes=probes)
        self.calls.append(dict(t=self.controls, obs=obs, noise=array_digest(noise), route=route,
                               history=history.tolist(), used=array_digest(history), proposal=array_digest(proposal),
                               proposed_prefix=proposal[:5].tolist()))
        return result


def cpu_fixture(fit, report):
    from cotrain_policy import CHECKPOINT
    from openpi.policies import policy_config
    from openpi.training import config
    from preflight import CPUFixture

    cfg = config.get_config("pi05_libero")
    with patch("openpi.models.model.restore_params", return_value={}), patch.object(
            type(cfg.model), "load", return_value=CPUFixture()):
        stock = policy_config.create_trained_policy(cfg, CHECKPOINT)
    with np.load(fit/"heads.npz", allow_pickle=False) as z:
        heads = {k: z[f"joint/{k}"] for k in report["parameters"]["joint"]["fitted"]}
    require({k: array_sha(v) for k, v in heads.items()} == report["parameters"]["joint"]["fitted"],
            "actual fitted CPU factors")
    native = NativeSampler(stock, {"joint": heads})
    return SimpleNamespace(stock=stock, _native=native)


def frozen(motor, fit, fixture):
    if not fixture:
        result = motor.digests()
    else:
        result = dict(resident_heads={"joint": {k: array_sha(v) for k, v in motor._native.heads["joint"].items()}},
                      resident_effective_kernels={"joint": array_sha(motor._native.kernels["joint"])})
    result["disk_fit"] = sha(fit/"fit.json")
    result["disk_heads"] = sha(fit/"heads.npz")
    return result


def run(out, pin, fit, manifest, fixture):
    reg, cases, tasks = matched.contract()
    require(sha(matched.HERE/"contract.json") == CONTRACT and sha(fit/"fit.json") == FIT, "frozen locks")
    require(jax.default_backend() == ("cpu" if fixture else "gpu") and len(jax.devices()) == 1, "device")
    if fixture:
        require(len(os.sched_getaffinity(0)) <= 2 and os.environ["CUDA_VISIBLE_DEVICES"] == "", "CPU budget")
    admission = inspect(fit, FIT, manifest)
    require(admission["source_manifest_sha256"] == reg["data_manifest"], "matched source manifest")
    report = json.loads((fit/"fit.json").read_text())
    require(sha(fit/"heads.npz") == report["heads_sha256"], "head archive")
    save(out/"run.json", dict(pin=pin, contract=CONTRACT, fit=str(fit), admission=admission, fixture=fixture,
                              affinity=sorted(os.sched_getaffinity(0)), host=socket.gethostname(), pid=os.getpid(),
                              jax=jax.__version__, devices=[str(d) for d in jax.devices()]))
    motor = cpu_fixture(fit, report) if fixture else LockedPolicy.load(fit)
    before = frozen(motor, fit, fixture)
    save(out/"frozen-before.json", before)
    engine = Sampler(motor)
    if not fixture:
        require(engine.kernel == report["parameters"]["joint"]["action_out_proj/kernel"], "live joint kernel")
    (out/"episodes").mkdir()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        sim = UR5eSimulator(allow_mesa=fixture)
    with sim, patch.object(rollout, "Wire", Wire), patch.object(verify, "TASKS", tasks):
        for case in cases:
            ref = None
            sim.call("create", dict(case, embodiment="nom"))
            for mode in MODES:
                v = matched.episode(sim, engine, case, mode, ref, fixture, reg["tasks"][case["suite"]])
                verify.validate(v, fixture)
                save(out/"episodes"/f"{key(case)}-{mode}.json", v)
                if ref is None:
                    ref = v
                print(key(case), mode, v["native"]["controls"], flush=True)
    after = frozen(motor, fit, fixture)
    save(out/"frozen-after.json", after)
    require(before == after, "frozen arrays")
    save(out/"graphs.json", engine.graph_ids)
    from audit import audit
    save(out/"paired.json", audit(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("pin")
    p.add_argument("fit", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("--fixture", action="store_true")
    run(**vars(p.parse_args()))
