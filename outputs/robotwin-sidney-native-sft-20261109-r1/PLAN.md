# Original Sidney SFT on native RoboTwin phone stand

Evaluate revision e49e2ab6c11f07511573b67261bd129e88d0a416 on the 100 clean and 100 randomized held-out seeds and unseen prompts saved in PHONE100-CASES.json. Use the restored native simulator at fa8d97b0 with its default D435 streams (320 × 240), per-target control, and 400-target episode limit. Check the prompt against the native unseen template and both actor IDs after each reset. Cases passed the previous native expert filter. Count every policy attempt. A setup exception interrupts the run.

The original LeRobot bundle and serialized processors are verified against the pinned manifest in the allocated job. The qualified batch-one eager worker uses absolute joint targets, K10 and H50, and verifies frozen parameters after every prediction. Noise seeds are fixed by case index and decision. Save reset inputs, actor IDs, predicted actions, delivered targets, outcomes and GPU binding. The first episode exercises the same path as the full panel. Results apply to these scene-specific instructions. The uploader's public LeRobot invocation defaults to a task-name prompt for phone stand.

One Baige Pro6K GPU with 370 minutes maximum (22,200 GPU-seconds) fits the existing 100 GPU-hour epoch.
