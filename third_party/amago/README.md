# Pinned AMAGO1 source

`upstream/amago/` is an unmodified copy of the official
[UT-Austin-RPL/amago](https://github.com/UT-Austin-RPL/amago/tree/514c1c2a471418aaf7829ba60ad0582b7764e643)
package at `514c1c2a471418aaf7829ba60ad0582b7764e643` (2024-03-18).
The original Meta-World example and `setup.py` are retained for provenance.
`SOURCE.json` hashes every included upstream file; the adapter checks them before
importing. The MIT license is retained in `LICENSE`; original source attribution
comments remain in place. Unneeded media, notebooks and unrelated examples are
omitted. No upstream algorithm file has been patched.

This is the original ICLR 2024 AMAGO agent, not AMAGO2's MultiTaskAgent or AMAGO3.
Upstream's `v2` Git tag already contains forthcoming multi-task changes and is
not used. METTA cites the ICLR 2024 paper but does not identify its AMAGO commit
or publish an exact AMAGO configuration. This pin is a transparent independent
baseline choice, not evidence of reproducing their baseline.

Use a separate Linux `uv` environment with `requirements.txt`, then install the
main repository with `--no-deps`. The adapter imports this source directly, so
do not install a different `amago` package. `requirements-flash.txt` lists the
optional exact FlashAttention2 build requirements. Their compatibility must be
qualified on the actual allocated GPU; preserve a compiled wheel and hash.

`requirements-sdpa.txt` is a separate native PyTorch SDPA profile pinned to
Torch 2.11.0+cu128 and NumPy 2.4.6. It requires no `flash-attn` package or CUDA
compiler. `ml10-amago1-sdpa.args` keeps every original model and optimizer
setting and explicitly selects the parameter-free SDPA primitive adapter.
This backend still needs hardware qualification; its native kernel dispatch
and timing must not be described as measurements of the external extension.

For the accepted one-seed comparison schedule, use
`ml10-amago1-sdpa-50env-10m-comparison.args`: 50 lanes, eight CPU workers within 16 allocated CPUs, K5H500, 80 rounds,
10M new training transitions, and ten passes' worth of sampled replay transitions
per collection. This explicitly changes AMAGO's training schedule. Cumulative
rounding gives 2083 default B24/L2000 optimizer steps and 99,984,000 sampled
transitions versus a 100,000,000 target. Warmup spans the first ten collection
rounds (260 optimizer steps). The separate `original-ratio.args` preset keeps
the original 1/12 optimizer updates per new transition and a 10,000-step warmup;
its much larger compute cost is not the primary comparison schedule.

When borrowing an existing qualified heavy runtime, `requirements-addon.txt`
lists only the five additional direct dependencies plus qualification pytest.
Install them in a new uv `--system-site-packages` overlay; the borrowed runtime
must remain read-only. Full runtime pins remain available as separate profiles.

`ml10-amago1.args` is consumed through argparse's `@file` syntax. See
[`docs/metaworld-amago.md`](../../docs/metaworld-amago.md) for commands,
scientific departures, checkpoint/replay recovery and admission requirements.
