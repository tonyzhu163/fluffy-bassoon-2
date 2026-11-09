# Original Sidney SFT with native RoboTwin camera settings

On 100 fixed phone-stand scenes per domain, the published LeRobot SFT weights scored **23/100 clean and 31/100 randomized**. The 95% Wilson intervals are 15.8–32.2% and 22.8–40.6%. The unmodified 10 cm wrist clipping distance hides the gripper fingertips on this SAPIEN 3.0.1 runtime. The paired camera experiment records performance after restoring them. [Camera experiment](../robotwin-sidney-gap-20261109-r1/RESULT.md).

The evaluator delivered 34,224 clean and 31,420 randomized targets through native control. The independent readback verified 1,340 saved 50-by-14 action chunks and all 200 renderer bindings. All 200 case seeds and prompts matched the frozen panel. Comparison with the earlier Motus run found byte-identical initial images, float32 state and prompt on all 100 clean and 98 randomized scenes. [Full evidence](READBACK.json).

The two randomized visual differences in the cross-model comparison occurred on seeds 100047 and 100073. All corresponding prompts and joint vectors matched. The Sidney clipping experiment compared its own two camera settings and obtained identical head-camera pixels on both of those seeds. [Per-camera measurements](RANDOMIZED-INPUT-COMPARISON.json).
