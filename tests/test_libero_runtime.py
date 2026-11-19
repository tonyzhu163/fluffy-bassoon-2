"""Unit tests for shared LIBERO timing, simulator reuse/chunk IPC, and transform caching."""

from contextlib import contextmanager
import importlib.util
from multiprocessing import Pipe
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

try:
    import pytest
except ImportError:  # Allow running the stdlib simulator IPC test without pytest.
    pytest = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "outputs" / "libero-meta-learning-20261117-r1"


def _load_module(name, path, extra_sys_path=None):
    added = []
    if extra_sys_path is not None and str(extra_sys_path) not in sys.path:
        sys.path.insert(0, str(extra_sys_path))
        added.append(str(extra_sys_path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for entry in added:
            if entry in sys.path:
                sys.path.remove(entry)


class _CopyList(list):
    def copy(self):
        return _CopyList(self)


@contextmanager
def _fake_libero_remote():
    state = {"manifest_loads": 0, "close_calls": 0, "create_calls": 0, "step_calls": 0}

    class FakeEnv:
        def __init__(self):
            self.active = False
            self.step_count = 0
            self.pos = [0.0, 0.0, 0.0]
            self.term_at = 999
            self.target = 999

        def create(self, spec):
            state["create_calls"] += 1
            self.active = True
            self.step_count = 0
            self.pos = [0.0, 0.0, 0.0]
            self.term_at = spec.get("term_at", 999)
            self.target = spec.get("target", 999)
            return {
                "observation": {"robot0_eef_pos": _CopyList(self.pos), "agentview_image": "init_img"},
                "language_instruction": "fake task",
                "environment_reproducibility": {"initial_state_fingerprint": f"fp-{state['create_calls']}"},
                "action_schema": {},
            }

        def step(self, action):
            state["step_calls"] += 1
            self.step_count += 1
            self.pos = [self.pos[i] + float(action[i]) for i in range(3)]
            done = self.step_count >= self.term_at
            success = self.step_count == self.target
            return {
                "observation": {
                    "robot0_eef_pos": _CopyList(self.pos),
                    "robot0_gripper_qpos": _CopyList([float(self.step_count), -float(self.step_count)]),
                    "agentview_image": f"img_{self.step_count}",
                },
                "reward": 1.0 if success else 0.0,
                "success": success,
                "terminated": done,
                "truncated": False,
            }

        def close(self):
            if self.active:
                state["close_calls"] += 1
                self.active = False

    class FakeManifest:
        @classmethod
        def load(cls, path):
            state["manifest_loads"] += 1
            return cls()

        def require(self, suite, task_id, variant):
            return {"max_steps": 50}

        def resolve_bddl(self, suite, task_id, variant):
            return Path("/tmp/fake.bddl")

    root_mod = ModuleType("libero_remote")
    env_pkg = ModuleType("libero_remote.environment")
    libero_mod = ModuleType("libero_remote.environment.libero")
    manifest_mod = ModuleType("libero_remote.environment.task_manifest")
    libero_mod.LiberoEnvironmentAdapter = FakeEnv
    manifest_mod.TaskManifest = FakeManifest
    injected = {
        "libero_remote": root_mod,
        "libero_remote.environment": env_pkg,
        "libero_remote.environment.libero": libero_mod,
        "libero_remote.environment.task_manifest": manifest_mod,
    }
    previous = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield state
    finally:
        for key, old in previous.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old


def test_estimate_command_lag_before_and_after_equivalence():
    np = pytest.importorskip("numpy")
    from fluffy_bassoon.libero import estimate_command_lag

    rng = np.random.default_rng(20261119)
    total = 90
    for lag in (0, 13):
        commands = rng.standard_normal((total, 7))
        positions = np.zeros((total + 1, 3), dtype=np.float64)
        for t in range(total):
            delta = commands[t - lag, :3] if t - lag >= 0 else np.zeros(3)
            positions[t + 1] = positions[t] + delta
        assert estimate_command_lag(commands, positions[:total], positions_timing="before") == lag
        assert estimate_command_lag(commands, positions[1 : total + 1], positions_timing="after") == lag

    # Short 24-step prefix with S20-like lag=14 and zero movement during initial latency.
    short_cmd = rng.standard_normal((24, 7))
    short_pos_all = np.zeros((25, 3), dtype=np.float64)
    for t in range(24):
        delta = short_cmd[t - 14, :3] if t >= 14 else np.zeros(3)
        short_pos_all[t + 1] = short_pos_all[t] + delta
    assert (
        estimate_command_lag(
            short_cmd,
            short_pos_all[1:],
            positions_timing="after",
            prefix=24,
            fixed_window=False,
        )
        == 14
    )

    # Degenerate, tied, and short-window edge cases return None.
    zeros = np.zeros((80, 3), dtype=np.float64)
    assert estimate_command_lag(zeros, zeros, positions_timing="before") is None
    periodic_cmd = np.tile([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], (40, 1))
    periodic_pos = np.cumsum(np.vstack([np.zeros((1, 3)), periodic_cmd]), axis=0)[:80]
    assert estimate_command_lag(periodic_cmd, periodic_pos, positions_timing="before") is None
    assert estimate_command_lag(zeros[:10], zeros[:10], positions_timing="after", fixed_window=True) is None

    with pytest.raises(ValueError, match="positions_timing"):
        estimate_command_lag(zeros, zeros, positions_timing="invalid")


def test_simulator_reuse_and_chunk_ipc():
    sim_mod = _load_module("libero_simulator_under_test", CAMPAIGN_DIR / "simulator.py", CAMPAIGN_DIR)
    parent, peer = Pipe()
    peer_fd = os.dup(peer.fileno())
    peer.close()
    with _fake_libero_remote() as state:
        worker = threading.Thread(target=sim_mod.serve, args=(peer_fd,), daemon=True)
        worker.start()
        try:
            parent.send(("create", {"suite": "libero_goal", "task_id": 1, "variant": "BASE", "term_at": 3, "target": 2}))
            ok, payload = parent.recv()
            assert ok, payload
            spec, first = payload
            assert spec["max_steps"] == 50 and first["observation"]["agentview_image"] == "init_img"

            # Backward-compatible single step.
            parent.send(("step", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            ok, step1 = parent.recv()
            assert ok, step1
            assert step1["observation"]["agentview_image"] == "img_1" and not step1["terminated"]

            # Batched chunk stops at step 3 (after 2 chunk steps) and returns only final observation.
            parent.send(("chunk", [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4))
            ok, chunk = parent.recv()
            assert ok, chunk
            assert chunk["steps"] == 2
            assert chunk["positions"] == [[1.5, 0.0, 0.0], [2.0, 0.0, 0.0]]
            assert chunk["gripper_positions"] == [[2.0, -2.0], [3.0, -3.0]]
            assert chunk["rewards"] == [1.0, 0.0]
            assert chunk["any_success"] is True and chunk["success"] is False
            assert chunk["terminated"] is True and chunk["truncated"] is False
            assert chunk["observation"]["agentview_image"] == "img_3"
            assert state["step_calls"] == 3

            # Second episode reuses manifest and closes prior env before creating new one.
            parent.send(("create", {"suite": "libero_spatial", "task_id": 1, "variant": "BASE"}))
            ok, payload = parent.recv()
            assert ok, payload
            _, second = payload
            assert second["environment_reproducibility"]["initial_state_fingerprint"] == "fp-2"
            assert state["manifest_loads"] == 1
            assert state["close_calls"] == 1
            assert state["create_calls"] == 2

            # Empty chunk fails gracefully and returns an error over the pipe.
            parent.send(("chunk", []))
            ok, err = parent.recv()
            assert not ok and "ValueError" in err
        finally:
            parent.close()
            worker.join(timeout=5)
            assert not worker.is_alive()

    # Verify Simulator context manager and idempotent close without spawning a real env.
    wait_calls = []

    class FakePopen:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            wait_calls.append(timeout)

        def terminate(self):
            raise AssertionError("terminate should not be called when wait succeeds")

    orig_popen = sim_mod.subprocess.Popen
    sim_mod.subprocess.Popen = FakePopen
    try:
        with sim_mod.Simulator() as sim:
            assert not sim._closed
            sim.close()
            assert sim._closed
        assert len(wait_calls) == 1
    finally:
        sim_mod.subprocess.Popen = orig_popen


def test_native_motor_caches_input_transform():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    native_mod = _load_module("libero_native_under_test", CAMPAIGN_DIR / "native.py")

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.transform_calls = 0
            self.fail_next = False

        def input_transform(self, data):
            self.transform_calls += 1
            return {
                "state_tensor": torch.ones((1, 8), dtype=torch.float32),
                "arr": np.ones((1, 7), dtype=np.float32),
                "nested": [torch.full((1, 2), 3.0, dtype=torch.float32)],
                "prompt": list(data["prompt"]),
            }

        def build_prefix_cache(self, observation):
            hidden = torch.ones((1, 4, 2048), dtype=torch.float32)
            mask = torch.ones((1, 4), dtype=torch.bool)
            return hidden, mask, ("k", "v")

        def predict_action_batch(self, item, *, mode, noise):
            if self.fail_next:
                raise RuntimeError("boom")
            transformed = self.input_transform(dict(prompt=item["task_descriptions"]))
            state = transformed["state_tensor"].clone()
            arr = transformed["arr"].copy()
            nested = transformed["nested"][0].clone()
            # Mutate in place to verify prepare/actions cloning protects cached transforms.
            transformed["state_tensor"].zero_()
            transformed["arr"].fill(99.0)
            transformed["nested"][0].fill_(-1.0)
            transformed["extra"] = "mutated"
            obs = SimpleNamespace(state=state)
            self.build_prefix_cache(obs)
            base = state[:, :7].unsqueeze(1) + torch.from_numpy(arr).unsqueeze(1) + nested[:, :1].unsqueeze(1)
            return base + noise[:, :10, :7], {}

    model = FakeModel()
    motor = native_mod.Motor.__new__(native_mod.Motor)
    motor.model = model
    item = {
        "main_images": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "states": np.zeros((1, 8), dtype=np.float32),
        "task_descriptions": ["put the bowl on the plate"],
    }

    prepared = motor.prepare(item)
    assert model.transform_calls == 1
    assert float(prepared["transformed"][0]["state_tensor"].sum()) == 8.0
    assert float(prepared["transformed"][0]["arr"].sum()) == 7.0

    noise = np.arange(320, dtype=np.float32).reshape(1, 10, 32) * 0.01
    expected = 5.0 + noise[0, :10, :7]
    for _ in range(3):
        actions = motor.actions(prepared, noise)
        assert model.transform_calls == 1
        np.testing.assert_allclose(actions, expected, atol=1e-6)

    # Restore original methods even when predict_action_batch fails.
    orig_prefix = model.build_prefix_cache
    orig_transform = model.input_transform
    model.fail_next = True
    with pytest.raises(RuntimeError, match="boom"):
        motor.actions(prepared, noise)
    assert model.build_prefix_cache is orig_prefix
    assert model.input_transform is orig_transform

    # Fallback when prepared lacks cached transformed outputs.
    model.fail_next = False
    legacy_prepared = {k: v for k, v in prepared.items() if k != "transformed"}
    legacy_actions = motor.actions(legacy_prepared, noise)
    assert model.transform_calls == 2
    np.testing.assert_allclose(legacy_actions, expected, atol=1e-6)
