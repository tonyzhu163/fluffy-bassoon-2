"""Single stock policy, file-bound fitted heads and read-only native flow sampling."""
import hashlib
import json
from pathlib import Path
import subprocess

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.model import Observation
from openpi.policies import policy_config
from openpi.training import config

from flow_head import ACTION_DIM, CONDITIONS, HISTORY_DIM, HORIZON, bind, effective_kernel, full_params, initial_params, live_sample

LEGACY = Path("/mowenjie/fluffy-bassoon")
CHECKPOINT = LEGACY / "inputs/pi05-libero-openpi-20261027-r1"
OPENPI = LEGACY / "third_party/openpi-215abfb"
REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
BASE_RECEIPT = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
NORM = "assets/physical-intelligence/libero/norm_stats.json"
SCHEMA = "pi05-libero-native-flow-v1"


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((str(a.dtype), a.shape)).encode() + b"\0")
    h.update(memoryview(a.view(np.uint8)))
    return h.hexdigest()


def base_identity(*, verify_files):
    if file_sha(CHECKPOINT / "READY.json") != BASE_RECEIPT:
        raise ValueError("Stock checkpoint manifest digest mismatch")
    receipt = json.loads((CHECKPOINT / "READY.json").read_text())
    for entry in receipt["files"]:
        f = CHECKPOINT / entry["name"]
        if f.stat().st_size != entry["bytes"]:
            raise ValueError(f"Stock checkpoint size mismatch: {f}")
        if (verify_files or entry["name"] == NORM) and file_sha(f) != entry["sha256"]:
            raise ValueError(f"Stock checkpoint digest mismatch: {f}")
    return {"openpi_revision": REVISION, "checkpoint_manifest": BASE_RECEIPT, "normalization": file_sha(CHECKPOINT / NORM)}


def load_stock():
    import openpi.models.pi0 as pi0
    if not Path(pi0.__file__).resolve().is_relative_to(OPENPI):
        raise RuntimeError(f"Unexpected OpenPI source {pi0.__file__}")
    rev = subprocess.check_output(["git", "-C", OPENPI, "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION or subprocess.call(["git", "-C", OPENPI, "diff", "--quiet", "HEAD", "--", "src", "packages"]):
        raise RuntimeError("OpenPI code differs from the pin")
    base_identity(verify_files=True)
    return policy_config.create_trained_policy(config.get_config("pi05_libero"), CHECKPOINT)


def transformed(stock, item):
    values = stock._input_transform(jax.tree.map(lambda x: x, item))
    stacked = jax.tree.map(lambda x: jnp.asarray(x)[None], values)
    return Observation.from_dict(stacked), stacked.get("actions")


def kernel_receipt(layer, params):
    return {"action_out_proj/kernel": array_sha(effective_kernel(layer.kernel.value, params)),
            "action_out_proj/bias": array_sha(layer.bias.value),
            "fitted": {k: array_sha(v) for k, v in sorted(params.items())}}


class LockedPolicy:
    def __init__(self, stock, fit_dir):
        self.stock = stock
        self.fit_dir = Path(fit_dir)
        report = json.loads((self.fit_dir / "fit.json").read_text())
        if (report["schema"] != SCHEMA or report["base"] != base_identity(verify_files=False)
                or report["heads_sha256"] != file_sha(self.fit_dir / "heads.npz")):
            raise ValueError("Fit identity mismatch")
        proj = stock._model.action_out_proj
        template = initial_params(proj.in_features, "joint")
        self._heads = {}
        with np.load(self.fit_dir / "heads.npz", allow_pickle=False) as archive:
            expected = set()
            for condition in CONDITIONS:
                params = {}
                for k, v in initial_params(proj.in_features, condition).items():
                    key = f"{condition}/{k}"
                    expected.add(key)
                    value = archive[key]
                    if value.shape != v.shape or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError(key)
                    params[k] = jnp.asarray(value)
                if kernel_receipt(proj, params) != report["parameters"][condition]:
                    raise ValueError(f"Incorrect fitted projection or branch: {condition}")
                self._heads[condition] = full_params(params, template)
            if set(archive.files) != expected:
                raise ValueError("Unexpected head arrays")
        self._sample = bind(stock._model, live_sample)

    @classmethod
    def load(cls, fit_dir):
        return cls(load_stock(), fit_dir)

    def digests(self):
        """Reread deployed arrays, on-disk fit and normalization. Streams one stock leaf at a time."""
        state = nnx.state(self.stock._model, nnx.Param).flat_state()
        resident = hashlib.sha256()
        for path, variable in sorted(state.items()):
            resident.update(json.dumps([path, array_sha(variable.value)]).encode())
        return {**base_identity(verify_files=False), "resident_stock": resident.hexdigest(),
                "resident_heads": {c: {k: array_sha(v) for k, v in p.items()} for c, p in self._heads.items()},
                "heads_file": file_sha(self.fit_dir / "heads.npz"), "fit_file": file_sha(self.fit_dir / "fit.json")}

    def _injected(self, item, noise, condition, history):
        obs, _ = transformed(self.stock, item)
        actions = self._sample(jax.random.key(0), obs, jnp.asarray(noise)[None],
                               self._heads[condition], jnp.asarray(history)[None])
        return self.stock._output_transform({"actions": np.asarray(actions[0]), "state": np.asarray(obs.state[0])})

    def infer(self, item, *, noise, condition, history):
        noise, history = np.asarray(noise), np.asarray(history)
        for x, shape in ((noise, (HORIZON, ACTION_DIM)), (history, (HISTORY_DIM,))):
            if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"Expected finite float32 {shape}, got {x.shape} {x.dtype}")
        if condition == "original":
            return self.stock.infer(item, noise=noise)
        if condition not in self._heads:
            raise ValueError(condition)
        return self._injected(item, noise, condition, history)

    def parity(self, item, *, noise):
        """Bitwise decoded-action check of stock inference against zero-update live injection."""
        stock = self.stock.infer(item, noise=noise)["actions"]
        live = self._injected(item, noise, "original", np.zeros(HISTORY_DIM, np.float32))["actions"]
        np.testing.assert_array_equal(stock, live)
        return {"shape": list(stock.shape), "stock_sha256": array_sha(stock), "noop_sha256": array_sha(live),
                "max_abs": float(np.max(np.abs(stock - live)))}
