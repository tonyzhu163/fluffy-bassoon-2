# Continuous production startup

Baige `job-fji2xx8oq53h` is running from 2026-10-31 02:26:01 Shanghai. Actual current-head/trained checkpoint admission, one-GPU CUDA probe, renderer binding and native noise parity (0.0 for both arms) passed. Runtime570 and the scientific DB lineage remain unchanged; qualified supervisor0c16 owns the new allocation.

The first finite source children continued actual R5 weights: PPO57→58,546458→556583source actions and285→290Adam steps; trained-reset59→60,564658→573979source actions and295→300Adam steps. Both completed all six K4 trials. Both children passed their allocated post-child recovery/raw/canonical audits and the supervisor launched each independent `slice-001` successor from its issued trained manifest. These are actual source updates, not inferred from collection.

Initial child outputs are `robotwin-trained-pair-initial-20261031-r1/{ppo,ppo_reset}`. Outer `ppo/slice-000` and `ppo_reset/slice-000` bind these outputs and logs; later children use their own `slice-NNN/experiment` directories. Read each `launch.json` and latest durable per-arm manifest rather than assuming all outputs sit under the supervisor root.

The original 100-round and final evaluation goals remain unfinished. The 13-hour scheduler boundary includes the initial 2700-second admission slice once. Per-arm progress is independent and source collection stops at100. No new pilot, architecture, action cap or failed automatic retry was added.

[Actual first updates](receipts/first-source-updates.json), [both post-child audits and successor launches](receipts/first-child-audits.json), [initial admission](receipts/initial-preflight.json), [CUDA probe](receipts/gpu-probe.json), [running scheduler](receipts/scheduler-running.json).
