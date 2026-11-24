"""Frozen π₀.₅ and two independently sealed, bias-free velocity adapters."""
import hashlib
import json
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared.array_typing import disable_typechecking

OUTPUTS = Path(__file__).resolve().parents[2]
PILOT = OUTPUTS / "frozen-vla-cotraining-20261124-r1"
for dependency in (PILOT / "model", PILOT / "data", OUTPUTS / "libero-xemb-ur5e-sysid-20261120-r1"):
    sys.path.insert(0, str(dependency))

from cotrain_policy import NativeSampler, array_sha, base_identity, file_sha, load_stock, transformed
from flow_head import history_offset

FITS = ("panda", "mixed")
SCHEMA = "pi05-plate-history-adapter-v1"


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, obj):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def state_digest(state):
    resident = hashlib.sha256()
    for path, var in sorted(state.flat_state().items()):
        resident.update(json.dumps([path, array_sha(var.value)]).encode())
    return resident.hexdigest()


def stock_digest(stock):
    return state_digest(nnx.state(stock._model, nnx.Param))


def head_digest(params):
    return {k: array_sha(v) for k, v in params.items()}


def read_head(directory, label, family):
    report = read_json(directory / "fit.json")
    if (file_sha(directory / "fit.json") != family["exports"][label]
            or report["schema"] != SCHEMA or report["condition"] != label
            or any(report[k] != family[k] for k in ("base", "code_revision", "data_seal_sha256", "resident_stock"))
            or file_sha(directory / "head.npz") != report["head_sha256"]):
        raise ValueError(f"Adapter artifact binding failed for {label}")
    with np.load(directory / "head.npz", allow_pickle=False) as z:
        params = {k: z[k] for k in z.files}
    shapes = {"history_in": (52, 16), "history_out": (16, 320)}
    if set(params) != set(shapes) or not all(
            p.shape == shapes[k] and p.dtype == np.float32 and np.isfinite(p).all() for k, p in params.items()):
        raise ValueError(f"Unexpected trained parameters for {label}")
    if head_digest(params) != report["parameters"]:
        raise ValueError(f"Adapter array digest failed for {label}")
    return jax.tree.map(jnp.asarray, params)


class PlatePolicy:
    def __init__(self, stock, heads, directory=None):
        self.stock = stock
        self.heads = dict(heads)
        self.heads["stock"] = jax.tree.map(jnp.zeros_like, heads["panda"])
        self.directory = directory
        self.native = NativeSampler(stock, {"stock": self.heads["stock"]})
        self.state = self.native.states["stock"]
        self.offset = jax.jit(history_offset)
        self.executables = {}
        self.graph_sha256 = {}

    @classmethod
    def restore(cls, stock, directory, family):
        if family["schema"] != SCHEMA or family["base"] != base_identity(verify_files=False):
            raise ValueError("Stock base or fit schema mismatch")
        return cls(stock, {c: read_head(directory / c, c, family) for c in FITS}, directory)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        family = read_json(directory / "fit.json")
        result = cls.restore(load_stock(), directory, family)
        if stock_digest(result.stock) != family["resident_stock"]:
            raise ValueError("Restored stock parameters differ from the source-fitted base")
        return result

    def digests(self):
        result = dict(base=base_identity(verify_files=False), resident_stock=stock_digest(self.stock),
                      deployed_stock=state_digest(self.state.filter(nnx.Param)),
                      resident_heads={k: head_digest(v) for k, v in self.heads.items()},
                      deployed_kernel=array_sha(self.native.kernels["stock"]))
        if result["resident_stock"] != result["deployed_stock"]:
            raise ValueError("Materialized sampler parameters differ from genuine stock")
        if self.directory is not None:
            result["files"] = {f: file_sha(self.directory / f) for f in (
                "fit.json", *(f"{c}/{f}" for c in FITS for f in ("fit.json", "head.npz")))}
        return result

    def _sample(self, item, noise, condition, history, graph):
        obs, _ = transformed(self.stock, item)
        noise = jnp.asarray(noise)[None]
        offset = self.offset(self.heads[condition], jnp.asarray(history)[None])
        args = self.state, jax.random.key(0), obs
        kw = {"noise": noise}
        native = self.native.plain
        if graph == "history":
            kw["offset"] = offset
            native = self.native.with_history
        if graph not in self.executables:
            with disable_typechecking():
                exe = native.lower(*args, **kw).compile()
            self.executables[graph] = exe
            self.graph_sha256[graph] = hashlib.sha256(exe.as_text().encode()).hexdigest()
        result = self.native.decode(obs, self.executables[graph](*args, **kw))
        return result, {"graph": graph, "hlo_sha256": self.graph_sha256[graph],
                        "offset_sha256": array_sha(offset), "condition": condition}

    def infer_with_receipt(self, item, *, noise, condition, history):
        for array, shape in ((noise, (10, 32)), (history, (52,))):
            x = np.asarray(array)
            if x.dtype != np.float32 or x.shape != shape or not np.isfinite(x).all():
                raise ValueError(f"Expected finite float32 array of shape {shape}")
        if condition not in self.heads:
            raise ValueError(condition)
        if condition == "stock":
            return self.stock.infer(item, noise=noise), {
                "graph": "stock_native", "hlo_sha256": None, "offset_sha256": None, "condition": condition}
        route = "history" if np.any(history) else "plain"
        return self._sample(item, noise, condition, history, route)

    def forced_zero(self, item, *, noise):
        """Unissued fixed-input diagnostic through the very same learned-offset executable."""
        return self._sample(item, noise, "stock", np.zeros(52, np.float32), "history")

    def infer(self, item, *, noise, condition, history):
        return self.infer_with_receipt(item, noise=noise, condition=condition, history=history)[0]

    def parity(self, item, *, noise):
        stock = self.stock.infer(item, noise=noise)["actions"]
        zero = np.zeros(52, np.float32)
        offsets, observed = {}, {}
        for c in self.heads:
            off = np.asarray(self.offset(self.heads[c], jnp.asarray(zero)[None]))
            if np.any(off) or not np.isfinite(off).all():
                raise AssertionError(f"Nonzero offset at empty history for {c}")
            offsets[c] = array_sha(off)
            a = self.infer(item, noise=noise, condition=c, history=zero)["actions"]
            if a.shape != (10, 7) or not np.isfinite(a).all() or array_sha(a) != array_sha(stock):
                raise AssertionError(f"Decoded opening differs from stock for {c}")
            observed[c] = array_sha(a)
        forced = self._sample(item, noise, "stock", zero, "history")[0]["actions"]
        return dict(stock_sha256=array_sha(stock), zero_actions_sha256=observed, zero_offsets_sha256=offsets,
                    zero_max_abs=0., forced_add_sha256=array_sha(forced),
                    forced_add_max_abs=float(np.abs(forced-stock).max()), graph_sha256=dict(self.graph_sha256))
