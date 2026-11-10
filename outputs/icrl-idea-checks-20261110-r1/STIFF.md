# Drawer damping and post-grasp translation

The frozen native policy lost all openings at 64× damping. Doubling its commanded translation after the first two-finger handle grasp left every paired native outcome unchanged. Stiffness affected success on these scenes, but this correction supplied no successful alternative at the highest damping.

| Damping scale | Measured slide damping | Normal effort | Doubled translation | Pairs with a two-finger grasp |
| ---: | ---: | ---: | ---: | ---: |
| 1× | 10 | 3/4 | 3/4 | 3/4 |
| 8× | 80 | 3/4 | 3/4 | 3/4 |
| 64× | 640 | 0/4 | 0/4 | 3/4 |

At 64×, the largest observed drawer travel was 0.048 m with normal effort and 0.193 m with doubled translation. The ordinary successful pulls reached 0.316–0.319 m. The two effort choices each opened 6 of 12 conditions. Choosing the better effort separately for every condition also opened 6 of 12.

The four pairs at each damping level use two development scenes crossed with two motor-noise seeds. All paired starts matched in simulator state and public observations. Within each effort pair, the first two-finger grasp matched in time and physical state, or both runs never grasped. The actuator multiplies the three translation commands after contact and clips to the controller bounds. It does not directly set a physical force. [Per-episode measurements](stiff02/events.jsonl).

Baige Pro6K job `job-19vkb4r2o74v` completed all 24 episodes and 10,430 controls on source `9c0036ea`. The worker used 941 seconds on one GPU (0.261 GPU-hours). [GPU probe](stiff02/gpu.json) · [Worker clock](stiff02/exit.json). The scheduler recorded 18m04s (0.301 GPU-hours) at release. [Scheduler receipt](stiff02-close.png).
