"""Single persistent simulator process for the receipt-bound target panel."""

import os
import subprocess
import sys
import traceback
from contextlib import ExitStack, contextmanager
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path

from large_transfer_stats import read, require, sha

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "libero-paligemma-crossarm-20261120-r1"))
PHYSICAL = (
    "Released grip-site pose unreachable",
    "Settling moved the paired task scene",
    "Task already successful at the first policy observation",
    "Post-settle scene differs from paired Panda",
)


@contextmanager
def worker(panel, plan_path):
    """Open the gated simulator only at the first reset on a cache miss."""
    with ExitStack() as stack:
        connection = None

        def call(op, value):
            nonlocal connection
            if connection is None:
                connection = stack.enter_context(_connection(panel, plan_path))
            return connection(op, value)

        yield call


@contextmanager
def _connection(panel, plan_path):
    from sim import _unpack

    channel, peer = Pipe()
    with channel, peer:
        proc = subprocess.Popen(
            [
                os.environ["CROSSARM_SIM_PYTHON"],
                "-Bu",
                str(Path(__file__).resolve()),
                str(peer.fileno()),
                str(panel),
                str(plan_path),
            ],
            pass_fds=(peer.fileno(),),
        )
        peer.close()
        try:

            def call(op, value):
                channel.send((op, value))
                status, result = channel.recv()
                if status == "error":
                    raise RuntimeError(result)
                if op == "advance":
                    obs, trace = result
                    return _unpack(obs), trace
                if status == "missing":
                    return None, result
                obs, receipt = result
                return _unpack(obs), receipt

            yield call
        finally:
            if proc.poll() is None:
                try:
                    channel.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def simulator_for(manifest):
    """Load the same reset and source-readback schema in the isolated simulator process."""
    from large_transfer_gate import PRODUCER, PRODUCER_IMPLEMENTATION, protocol_for
    from large_transfer_stats import fingerprint
    from types import ModuleType
    import hashlib

    protocol = protocol_for(manifest)
    if fingerprint(manifest["implementation"]) != PRODUCER_IMPLEMENTATION:
        import large_panel_sim
        return large_panel_sim
    name = "large_panel_sim.py"
    path = HERE / name
    checksum = manifest["implementation"][name]
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != checksum:
        source = subprocess.check_output(["git", "-C", str(HERE), "show",
                                          f"{PRODUCER}:outputs/{HERE.name}/{name}"])
    require(hashlib.sha256(source).hexdigest() == checksum, "Changed registered simulator")
    # This process serves one immutable panel. The original bridge imports its
    # own tape reader by this name and keeps its own per-layout admission set.
    sys.modules["large_panel_protocol"] = protocol
    module = ModuleType("registered_large_panel_sim")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def serve(fd, root, plan):
    import numpy as np
    from sim import Session, _pack

    from generated_layouts import LayoutRejected

    with Connection(fd) as channel:
        try:
            target = read(plan)
            simulator = simulator_for(read(root / "test" / "manifest.json"))
            rows = simulator.target_gate(root, "test")["layouts"]
            layouts = target["layouts"]
            require(len(rows) == len(layouts) == 100, "Incomplete native target panel")
            with Session("UR5e") as env:
                while True:
                    op, value = channel.recv()
                    if op == "close":
                        return
                    if op == "reset":
                        row = layouts[value]
                        try:
                            record = rows[value]
                            path = root / "test" / row["state_file"]
                            require(
                                record["state_file"] == row["state_file"]
                                and record["file_sha256"] == row["file_sha256"]
                                and record["initial_state_sha256"] == row["row"]
                                and record["movable_xy_hash"] == row["xy"]
                                and sha(path) == row["file_sha256"]
                                and record["layout_seed"] == row["seed"]
                                and record["scene_hash"] == row["scene"],
                                "Changed query layout",
                            )
                            obs, info, _ = simulator.physical_reset(
                                env,
                                "UR5e",
                                True,
                                np.load(path, allow_pickle=False),
                                record,
                            )
                        except ValueError as exc:
                            if not isinstance(exc, LayoutRejected) and not str(
                                exc
                            ).startswith(PHYSICAL):
                                raise
                            channel.send(("missing", str(exc)))
                        else:
                            channel.send(("ok", (_pack(obs), info)))
                    elif op == "advance":
                        require(1 <= len(value) <= 5, "Invalid prefix")
                        trace = []
                        for action in value:
                            obs, motion, reward, done, success = env.step(action)
                            trace.append((motion.tolist(), reward, done, success))
                            if done or success:
                                break
                        channel.send(("ok", (_pack(obs), trace)))
                    else:
                        raise ValueError(f"Unknown request {op}")
        except Exception:  # noqa: BLE001 -- propagate simulator errors across the interpreter boundary
            channel.send(("error", traceback.format_exc()))


if __name__ == "__main__":
    serve(int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
