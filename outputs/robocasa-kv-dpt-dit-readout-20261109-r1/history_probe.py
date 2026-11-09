"""Controlled CPU intervention in the two actual validation-selected KV policies."""
import json
import sys
from pathlib import Path

import torch
from fluffy_bassoon.meta_rl.flow_history import load_bridge_policy

HASHES = {
    "dpt": "7a0c4f6c3fbb92f54c3b6f0d9a36828e8d16fce2e5970b73a23ce77e8286f217",
    "dit": "7ccaae7048df38a90ced30530002469f1f73e1ee9cb37a17b07c8137cabb3721",
}


@torch.inference_mode()
def run(fits, method):
    policy, manifest = load_bridge_policy(fits / method / "candidate.pt", expected_sha256=HASHES[method])
    cfg = policy.config
    assert manifest["contract"]["objective"] == method and cfg.interface == "kv_bridge"
    assert not any(p.requires_grad for p in policy.parameters())
    gen = torch.Generator().manual_seed(64105000)
    token = torch.zeros(1, policy.core.input_dim)
    d = cfg.observation_dim
    state = None
    for step in range(12):
        token[:, :d] = policy.observation_mean + .25 * policy.observation_std * torch.randn(1, d, generator=gen)
        token[:, -1] = float(step == 0)
        _, _, state = policy.step(token, state)
    query = torch.zeros_like(token)
    query[:, :d] = policy.observation_mean
    query[:, -1] = 1
    def context(st, reset=False):
        control, _, after = policy.step(query, st, reset=torch.tensor([reset]))
        return control[:, :cfg.width], int(after["length"][0])
    full, nfull = context(state)
    reset, nreset = context(state, True)
    fresh, nfresh = context(None)
    assert (nfull, nreset, nfresh) == (13, 1, 1) and torch.equal(reset, fresh)
    mask = torch.tensor([[True, False]])
    def kv():
        return torch.randn(1, 2, cfg.kv_heads, cfg.head_dim, generator=gen)
    cache = tuple((kv(), kv()) for _ in range(18))
    a, b = (policy.kv_adapter(cache, mask, h) for h in (full, reset))
    deltas = []
    for av, bv, orig in zip(a, b, cache):
        for x, y, z in zip(av, bv, orig):
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            assert torch.equal(x[~mask], z[~mask]) and torch.equal(y[~mask], z[~mask])
        deltas.append(max(float((x[mask]-y[mask]).abs().max()) for x, y in zip(av, bv)))
    assert all(delta > 0 for delta in deltas)
    return dict(candidate_sha256=HASHES[method], history_queries=[nfull,nreset,nfresh],
                reset_exactly_matches_fresh=True, frozen=True, changed_layers=len(deltas),
                maximum_history_vs_reset_kv_delta=max(deltas), context_max_abs_delta=float((full-reset).abs().max()))

if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(dict(test="same synthetic current token and cache; 12 prior queries vs explicit reset",
                          device="cpu", simulator_episodes=0,
                          methods={m: run(Path(sys.argv[1]), m) for m in HASHES}), indent=2))
