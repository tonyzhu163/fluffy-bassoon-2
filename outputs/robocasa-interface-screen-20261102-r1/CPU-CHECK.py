"""Run numerical interface checks on allocated Linux CPU; no robot episodes."""
import importlib.util
from pathlib import Path
import platform

assert platform.system() == 'Linux'
import torch

spec = importlib.util.spec_from_file_location('screen_interfaces', Path(__file__).with_name('INTERFACES.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
torch.manual_seed(12)
cache = ((torch.randn(2, 4, 9, 16), torch.randn(2, 4, 9, 16)),
         (torch.randn(2, 4, 9, 16), torch.randn(2, 4, 9, 16)))
zero, code = torch.zeros(2, 32), torch.randn(2, 32) * .5
assert module.kv_delta(cache, zero, .1) is cache
changed = module.kv_delta(cache, code, .1)
assert changed[0] is cache[0] and changed[-1][0] is cache[-1][0]
assert not torch.equal(changed[-1][1], cache[-1][1])
assert torch.isfinite(changed[-1][1]).all()
velocity = torch.randn(2, 50, 32)
assert module.velocity_delta(velocity, zero, .25) is velocity
torch.testing.assert_close(module.velocity_delta(velocity, code, .25) - velocity,
                           (.25 * torch.tanh(code))[:, None, :].expand(-1, 50, -1), atol=2e-7, rtol=1e-5)
head = module.DirectHead().eval()
features = torch.randn(2, 2088)
altered = features.clone(); altered[:, 16:40] += 1000
assert torch.equal(head(features), head(altered))
assert head(features).shape == (2, 50, 12)
matrix = module.projection(32, 16, 921003)
assert torch.equal(matrix, module.projection(32, 16, 921003))
print('PASS: KV zero identity/nonzero value-only intervention; velocity reference; direct base-action independence; projection reproducibility')
