"""Small deterministic CPU motor with a visible current-history intervention."""
import numpy as np
from ur5e_motor import array_digest

class Fixture:
    def __init__(self):
        self.weights = np.array([0.03, 0.12], np.float32)

    def stock_actions(self, item, noise):
        actions = self.weights[0] * noise[:, :7]
        actions[:, 0] += item["observation/image"].mean() / 1500
        actions[:, 1] += item["observation/wrist_image"].mean() / 1400
        actions[:, 2] += np.tanh(item["observation/state"].sum()) / 12
        return actions

    def infer(self, item, *, noise, condition, history):
        a = self.stock_actions(item, noise)
        if condition in ("policy_only", "joint"):
            a[:, 3] += self.weights[1]
        if condition in ("adapter_only", "joint"):
            a[:, 2] += self.weights[1] * history[-1]
        return {"actions": a}

    def parity(self, item, *, noise):
        a = self.stock_actions(item, noise)
        b = self.infer(item, noise=noise, condition="adapter_only", history=np.zeros(52, np.float32))["actions"]
        np.testing.assert_array_equal(a, b)
        return dict(stock_sha256=array_digest(a), noop_sha256=array_digest(b), shape=list(a.shape), max_abs=0.0)

    def digests(self):
        return dict(fixture=array_digest(self.weights))

def native_timeout(sim):
    """Exercise the unchanged runner through the official drawer horizon on a historical, unscored start."""
    from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
    from ur5e_motor import run_episode
    from panel import require

    class Stationary:
        def infer(self, item, *, noise):
            return {"actions": np.zeros((10, 7), dtype=np.float32)}

    case = dict(arm="Kinova3", suite="libero_goal", task_id=3, variant="BASE", init_state_id=14, seed=41810314)
    result = run_episode(sim, Stationary(), case, get_embodiment_spec("nom"))
    require(result["controls"] == result["reset"]["horizon"] == 300 and result["truncated"]
            and not result["success"] and result["native_rewards"] == [0.0]*300, "native timeout fixture")
    return dict(case=case, stationary_controls=result["controls"], truncated=result["truncated"],
                native_rewards=result["native_rewards"], renderer=result["reset"]["renderer"])
