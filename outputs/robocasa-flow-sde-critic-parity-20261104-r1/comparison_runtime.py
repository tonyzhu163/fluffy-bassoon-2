"""Read-only first-microbatch value gate around the frozen historical learner."""
import os
from pathlib import Path

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import FlowActor


class ComparisonActor(FlowActor):
    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        if self._flow_first_micro_checked:
            return super().train_micro_batch(micro_batch, metrics, is_last=is_last)

        def check(module, inputs, output):
            new = output['values'].detach().float().flatten()
            old = micro_batch['prev_values'].detach().to(new.device).float().flatten()
            assert new.shape == old.shape
            delta = (new - old).abs()
            finite = bool(torch.isfinite(delta).all())
            passed = finite and float(delta.max()) <= 1e-3
            report = dict(passed=passed, finite=finite, tolerance=1e-3,
                max_abs_value_delta=float(delta.max()), mean_abs_value_delta=float(delta.mean()),
                before_backward=True, optimizer_steps=self.optimizer_steps,
                arm=os.environ['CRITIC_COMPARISON_ARM'])
            write(Path(self.cfg.native_pilot.output) / 'value-parity' /
                f'iteration-{self.version + 1:03d}-rank-{self._rank:02d}.json', report)
            if os.environ['CRITIC_COMPARISON_ARM'] == 'fixed' and not passed:
                raise RuntimeError(f'Fixed unchanged-policy value parity failed: {report}')

        hook = self.model.register_forward_hook(check)
        try:
            return super().train_micro_batch(micro_batch, metrics, is_last=is_last)
        finally:
            hook.remove()
