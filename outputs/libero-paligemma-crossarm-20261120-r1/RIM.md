# Sawyer Goal 5 inner-rim push

The rear approach stalled with the fingers at 1.087 m over a 0.9 m table. This controller closes the jaws, lifts, centers above the plate, and descends there. Once the fingers touch the plate or reach the table clearance, it sweeps toward the goal-side inner lip. It holds the measured seating height with 2 mm of downward preload, bounded to 1 mm above the table. The destination matches the existing direct controller.

For a fresh Sawyer source reset, import `RimExpert` from `rim_expert` and instantiate `expert = RimExpert(scene)` in the optional route of `demos.collect_attempt`. Its existing loop can use `expert.points`, log `expert.phase`, issue `env.step(expert.action(scene))`, and refresh `scene = env.expert_scene()` after every step.

Native qualification is pending. Start with Sawyer pose 0 and the same seed as the failed rear approach. Inspect the transition from `lower` to `sweep`, finger–plate contact, and plate displacement toward the goal. A high stall before `sweep` challenges the clearance hypothesis. Reaching the plate and sweeping across it without moving its center challenges the inner-rim purchase hypothesis. Accept a tape only on native terminal success with positive reward.
