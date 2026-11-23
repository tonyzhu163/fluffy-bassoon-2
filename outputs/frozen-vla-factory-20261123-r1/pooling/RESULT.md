# Pool three motion chunks for passive UR5e yaw

On the 64 held-out interfaces, equal pooling over three motion chunks gave the learned reader 6.53° mean absolute yaw error. Reading only the final chunk cost 5.52° (95% task-bootstrap interval +3.12° to +8.25°). Reading only the first cost 4.28° (+2.84° to +6.11°). The linear reader also benefited from using all three.

| Readout | Learned visual MAE | Linear visual MAE |
| --- | ---: | ---: |
| Equal circular mean | 6.53° | 9.16° |
| Motion-magnitude-weighted mean | 6.18° | 9.26° |
| Circular medoid | 6.26° | 9.01° |
| First chunk | 10.81° | 11.36° |
| Last chunk | 12.05° | 16.56° |
| Identity correction | 101.25° | 101.25° |

The primary paired change for magnitude weighting was −0.36° (−0.81° to +0.03°) with the learned reader and +0.10° (−0.33° to +0.57°) with the linear reader. This panel supports retaining temporal pooling. It leaves the choice of magnitude versus equal weights unresolved. Each baseline reproduced its archived equal-weight prediction on all 96 supports. The eight task identities supply the bootstrap clusters. The experiment reused the frozen policy’s support trajectories and visual vectors. It collected zero new episodes.

From the repository root, run each command with the pinned project environment.

```bash
uv run --no-sync python outputs/frozen-vla-factory-20261123-r1/pooling/pool.py learned
uv run --no-sync python outputs/frozen-vla-factory-20261123-r1/pooling/pool.py linear
```

[Plan and data revision](PLAN.md) · [Learned scores](learned.json) · [Linear scores](linear.json) · [Source evidence](../../symmeta-matched-flow-20261123-r1/RESULT.md)
