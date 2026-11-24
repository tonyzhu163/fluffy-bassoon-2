"""Use the qualified episode driver and its original compiled history sampler."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import inspect
from pathlib import Path

from jax._src import compiler
import numpy as np

import panel
from large_panel_protocol import require
from plate_policy import PlatePolicy, array_sha

MODES = ("stock", "mixed_on", "mixed_zero")
OFFSET_SHAPE = (1, 10, 32)
ZERO_SHA = array_sha(np.zeros(OFFSET_SHAPE, np.float32))


@contextmanager
def observe_stock_compile(record):
    """Record the executable JAX returns to the unmodified published sampler."""
    original = compiler.compile_or_get_cached

    def compile_and_record(*args, **kwargs):
        executable = original(*args, **kwargs)
        record.append("\n\n".join(m.to_string() for m in executable.hlo_modules()))
        return executable

    compiler.compile_or_get_cached = compile_and_record
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


class ZeroBranchPolicy(PlatePolicy):
    graph_directory = None

    def __init__(self, stock, heads, directory=None):
        super().__init__(stock, heads, directory)
        self.stock_jit = inspect.getclosurevars(stock._sample_actions).nonlocals["jitted_fn"]
        self.stock_hlo = None
        self.history_executable = None

    def _archive(self, name, hlo):
        with gzip.open(self.graph_directory / f"{name}.hlo.gz", "xb", compresslevel=1) as file:
            file.write(hlo.encode())
        return hashlib.sha256(hlo.encode()).hexdigest()

    def _stock_with_hlo(self, item, noise):
        if self.stock_hlo is None:
            records = []
            require(self.stock_jit._cache_size() == 0, "published sampler had already compiled")
            with observe_stock_compile(records):
                result = self.stock.infer(item, noise=noise)
            program, = (text for text in records if text.startswith("HloModule jit_fun,"))
            self.stock_hlo = self._archive("stock", program)
        else:
            result = self.stock.infer(item, noise=noise)
        require(self.stock_jit._cache_size() == 1, "published sampler signature changed")
        return result

    def infer_with_receipt(self, item, *, noise, condition, history):
        require(condition in MODES, "unknown experimental arm")
        for x, shape in ((noise, (10, 32)), (history, (52,))):
            require(x.shape == shape and x.dtype == np.float32 and np.isfinite(x).all(), "bad sampler input")
        if condition == "stock" or not np.any(history):
            result = self._stock_with_hlo(item, noise)
            return result, dict(graph="stock_native", hlo_sha256=self.stock_hlo, offset_sha256=None,
                                intervention="none", condition=condition)
        if condition == "mixed_on":
            result, route = self._sample(item, noise, "mixed", history, "history")
        else:
            # The qualified forced-zero helper supplies positive float32 zeros as an
            # ordinary offset array to the very same cached history executable.
            result, route = self.forced_zero(item, noise=noise)
            require(route["offset_sha256"] == ZERO_SHA, "forced offset is not numerical zero")
        executable = self.executables["history"]
        if self.history_executable is None:
            offset = executable.args_info[1]["offset"]
            require(offset.shape == OFFSET_SHAPE and offset.dtype == np.float32, "offset is not a dynamic float32 input")
            self.history_executable = executable
            require(self._archive("history", executable.as_text()) == route["hlo_sha256"],
                    "history HLO does not reproduce the issued executable digest")
        require(self.history_executable is executable, "history executable changed")
        return result, dict(route, condition=condition, offset_shape=list(OFFSET_SHAPE), offset_dtype="float32",
                            intervention="learned" if condition == "mixed_on" else "zero")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    for k in ("fit", "manifest"):
        parser.add_argument(f"--{k}", type=Path)
    for k in ("fit-pin", "fit-sha", "manifest-sha"):
        parser.add_argument(f"--{k}", required=True)
    for k in ("start", "stop", "shard"):
        parser.add_argument(f"--{k}", type=int, required=True)
    args = parser.parse_args()
    args.fixture = False
    args.resume = None
    ZeroBranchPolicy.graph_directory = args.out
    # panel.run resolves this class at runtime. All I/O, history updates, pairing and
    # completion seals stay in the qualified driver.
    import plate_policy
    plate_policy.PlatePolicy = ZeroBranchPolicy
    panel.MODES = MODES
    panel.run(args)


if __name__ == "__main__":
    main()
