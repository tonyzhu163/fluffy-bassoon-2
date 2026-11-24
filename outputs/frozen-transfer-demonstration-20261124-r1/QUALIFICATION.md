# CPU qualification

Baige development attempt `cpu02` passed at detached revision `ef165714d626305fa756e10fac2b5f7c45446f1c`, using cores 38/39, CUDA hidden, JAX CPU and native Mesa rendering. Four matched openings on reused Kinova3 layout 40 produced 240 physical controls. Raw and tuned each ran five controls with the fixture proposal. Both hybrids used the real frozen π0.5 checkpoint on the first post-cut observation, with the original instruction and global-index noise. The fixture supplied the earlier proposals during donor execution. [Qualification](evidence/cpu02/qualified.json) · [Origin](evidence/cpu02/origin.json) · [Full native fixture episodes](evidence/cpu02/episodes.tar.gz)

| Donor | Source controls issued | First frozen replan (zero-based) | First native control from frozen motor (one-based) | Physical controls consumed | Max. absolute channel change in first post-cut action |
|---|---:|---:|---:|---:|---:|
| IIWA 34 | 105 | 21 | 106 | 110 | 0.038340 |
| IIWA 35 | 115 | 23 | 116 | 120 | 0.033804 |

The reducer matched the underlying native step to all 220 source controls byte for byte and to the independently reconstructed tuned transform for all ten subsequent controls. Each first frozen query received a changed target observation whose digest matched the last donor step response. The opening reset, camera/proprioceptive inputs, proposal and noise matched across the quartet. No fixture episode reached a native terminal. [Boundary values](evidence/cpu02/boundaries.json) · [Step and noise readback](evidence/cpu02/readback.json) · [Exit](evidence/cpu02/exit.txt)

The earlier cpu01 diagnostic on `e178ee73462b` also passed both handoffs. Before dispatch a seeded Linux float32 fixture showed that transforming an already-shortened terminal chunk could differ by one ULP from the true motor's full five-row transform. The final verifier transforms the five proposals first and compares the consumed rows. The final writer saves the original episode before checking it. CPU02 repeated qualification on this revision. [Earlier diagnostic](evidence/cpu01/qualified.json) · [Pinned launch](HANDOFF.md)
