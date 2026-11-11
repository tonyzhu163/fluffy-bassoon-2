"""Numerical run qualification; execute only inside the allocated Linux job."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

code, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(code / "exploration/learning-histories"))
import sessions as S
from train import Learner, logits

assert torch.cuda.is_available()
assert not set(S.TRAIN) & set(S.HELDOUT)
s = S.sample(np.random.default_rng(7), 32, 6, "train")
c, o = S.behave(np.random.default_rng(8), s, 6)
model = Learner(8).cuda().eval()
errors = []
with torch.no_grad():
    for cond in S.CONDS:
        x, mask = S.encode(s, cond, c, o, 3, 7)
        assert not x[:, 7, :, S.A:S.A+4].any()
        assert mask[:, 4:7].all() and not mask[:, 7].any()
        full = logits(model, "cuda", s, cond, c, o, 3, 7)
        parts = []
        for lo in range(0, 32, 8):
            sl = slice(lo, lo+8)
            chunk = S.Sessions(s.rule[sl], s.scenes[sl], s.tgt[sl], s.true[sl])
            parts.append(logits(model, "cuda", chunk, cond, c[sl], o[sl], 3, 7))
        chunked = torch.cat(parts)
        torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-6)
        assert torch.equal(full.argmax(1), chunked.argmax(1))
        errors.append(float((full-chunked).abs().max()))

base = [sys.executable, "-Bu", str(code / "exploration/learning-histories/train.py"),
        "--cond", "attempts", "--K", "6", "--steps", "40", "--batch", "16",
        "--eval-n", "32", "--eval-batch", "8", "--device", "cuda"]
subprocess.run(base + ["--out", str(out/"whole.json")], check=True)
subprocess.run(base + ["--out", str(out/"partial.json"), "--stop-after", "20"], check=True)
subprocess.run(base + ["--out", str(out/"resumed.json"), "--resume", str(out/"partial.pt")], check=True)
a = torch.load(out/"whole.pt", map_location="cpu", weights_only=False)
b = torch.load(out/"resumed.pt", map_location="cpu", weights_only=False)

def equal(x, y):
    if isinstance(x, torch.Tensor):
        assert torch.equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            equal(x[key], y[key])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for left, right in zip(x, y):
            equal(left, right)
    else:
        assert x == y

for key in ("step", "model", "optimizer", "scheduler", "numpy_rng", "torch_rng", "cuda_rng"):
    equal(a[key], b[key])
whole, resumed = (json.loads((out/name).read_text()) for name in ("whole.json", "resumed.json"))
for split in ("train", "heldout"):
    assert whole[split] == resumed[split]
(out/"qualification.json").write_text(json.dumps(dict(status="passed", device=torch.cuda.get_device_name(),
    torch=torch.__version__, numpy=np.__version__, checkpoint_continuation="bitwise_equal",
    eval_chunk_max_abs_errors=errors), indent=2)+"\n")
print("QUALIFICATION PASSED", flush=True)
