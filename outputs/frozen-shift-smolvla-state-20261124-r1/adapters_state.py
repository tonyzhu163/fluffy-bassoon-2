"""State-branch adapters for the SmolVLA harness. Select with HOOKS=adapters_state:Name."""
import numpy as np

from adapters import LiberoPrompt


def other_branch(state):
    """Same rotation written as the other axis-angle branch: r -> r (theta - 2 pi) / theta."""
    out = np.array(state, dtype=np.float64)
    theta = np.linalg.norm(out[3:6])
    if theta < 1e-6:
        raise ValueError("Rotation too small to have a second branch")
    out[3:6] *= (theta - 2 * np.pi) / theta
    return out.astype(np.float32)


class Raw(LiberoPrompt):
    name = "state-raw"


class SourceBranch(LiberoPrompt):
    """Kinova3: put a negative-X rotation vector on the positive-X branch used by the source arms."""
    name = "state-source-branch"

    def state(self, state8, obs, context):
        return other_branch(state8) if state8[3] < 0 else state8


class TargetBranch(LiberoPrompt):
    """Panda control: put a positive-X rotation vector on the negative-X branch Kinova3 reports."""
    name = "state-target-branch"

    def state(self, state8, obs, context):
        return other_branch(state8) if state8[3] > 0 else state8
