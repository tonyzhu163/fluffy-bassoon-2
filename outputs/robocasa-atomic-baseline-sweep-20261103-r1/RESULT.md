# RoboCasa365 atomic_seen native baseline sweep — results

Completed 2026-11-03 on Baige Pro6K, job `job-jnlxhvwhrdji`; source `42cefb77daec997df5b6bd178aaf8e4bdfb3b089`.

All **240/240** planned episodes completed (20 per task), with **147,000 physical controls, zero unresolved intents and zero optimizer updates**. All eight workers exited successfully; the scheduler reported success and released the allocation at **15:52:35 Shanghai**. Conservative lifetime charge: **6,328 GPU-seconds (1.758 GPU-hours)**, including startup and release, within 26,400 GPU-seconds.

## Per-task results

First success means the native success condition held at least once during the complete episode; end success means it held at the final control. The table reports successes out of 20, rate and Wilson 95% interval. Categories use the first-success point estimate, as a coarse screen.

| Task | Horizon | First success /20 (rate; Wilson 95%) | End success /20 (rate; Wilson 95%) | Screen |
| --- | ---: | --- | --- | --- |
| CloseBlenderLid | 900 | 1/20 (5%; 0.9–23.6%) | 1/20 (5%; 0.9–23.6%) | 5–70% |
| CloseToasterOvenDoor | 450 | 5/20 (25%; 11.2–46.9%) | 5/20 (25%; 11.2–46.9%) | 5–70% |
| OpenCabinet | 1050 | 15/20 (75%; 53.1–88.8%) | 15/20 (75%; 53.1–88.8%) | above 70% |
| OpenStandMixerHead | 450 | 13/20 (65%; 43.3–81.9%) | 13/20 (65%; 43.3–81.9%) | 5–70% |
| SlideDishwasherRack | 450 | 14/20 (70%; 48.1–85.5%) | 13/20 (65%; 43.3–81.9%) | 5–70% |
| PickPlaceCounterToStove | 600 | 10/20 (50%; 29.9–70.1%) | 10/20 (50%; 29.9–70.1%) | 5–70% |
| PickPlaceSinkToCounter | 900 | 19/20 (95%; 76.4–99.1%) | 19/20 (95%; 76.4–99.1%) | above 70% |
| PickPlaceToasterToCounter | 600 | 15/20 (75%; 53.1–88.8%) | 15/20 (75%; 53.1–88.8%) | above 70% |
| TurnOnElectricKettle | 450 | 8/20 (40%; 21.9–61.3%) | 8/20 (40%; 21.9–61.3%) | 5–70% |
| TurnOnMicrowave | 450 | 6/20 (30%; 14.5–51.9%) | 4/20 (20%; 8.1–41.6%) | 5–70% |
| TurnOnSinkFaucet | 600 | 10/20 (50%; 29.9–70.1%) | 10/20 (50%; 29.9–70.1%) | 5–70% |
| NavigateKitchen | 450 | 2/20 (10%; 2.8–30.1%) | 2/20 (10%; 2.8–30.1%) | 5–70% |

## Implications for task selection

**PickPlaceCounterToStove (10/20, 50%) is the only newly screened pick-and-place task inside the 5–70% point-estimate range.** PickPlaceSinkToCounter (19/20, 95%) and PickPlaceToasterToCounter (15/20, 75%) exceed it; OpenCabinet also scored 15/20 (75%). CloseToasterOvenDoor (25%), TurnOnElectricKettle (40%), TurnOnMicrowave (30%), TurnOnSinkFaucet (50%), OpenStandMixerHead (65%) and SlideDishwasherRack (70%) offer additional coarse candidates.

SlideDishwasherRack lost success before the end in one episode (14 first / 13 end); TurnOnMicrowave did so in two (6 first / 4 end). Keep these metrics distinct. CloseBlenderLid is at the lower boundary (1/20, 5%); NavigateKitchen is reported separately (2/20, 10%) because navigation is a different task family.

This small single-bank screen does not qualify a final 5 ID + 5 OOD portfolio. The human selects that portfolio; selected tasks without an existing matched 50-episode qualification need that confirmation before PPO. The six previously measured tasks were not rerun, and historical native horizons must remain explicit when comparing them with later 600-control results.

## Verification and retained evidence

- [Independent final audit](FINAL-AUDIT.json): all 240 exact case seeds, all 2,940 motor-noise schedules, every zero control and base/executed chunk equality, and all simulator sidecars checked.
- [Video audit](VIDEO-AUDIT.json): all 720 videos fully decoded, 441,720 frames total, each 256×256 with exactly horizon+1 frames. The evaluator verified every saved video hash.
- [Per-case results and video hashes](CASE-RESULTS.json), [machine-readable aggregate](RESULT.json), [terminal accounting](TERMINAL-ACCOUNTING.json), [process clock](PROCESS-CLOCK.json).
- [Linux CPU qualification](QUALIFIED-LATEST.json), [29 passing core tests](CORE-TESTS.txt), [eight-device in-job probe](GPU-PROBE.json).
- Runtime: unchanged released Human300 pi0.5, H50 / 10 flow steps / three views; four concurrent simulator lanes per GPU. Native inference uses BF16 batching, whose outputs need not be bitwise equal to singleton inference. Each executed action chunk exactly matched its own native base chunk.
- Raw videos, saved per-episode chunk records, worker logs and action journals remain under `/mowenjie/fluffy-bassoon-2/outputs/robocasa-atomic-baseline-sweep-20261103-r1/gpu-r1`; approximately 1.50 GB. No active job or successor allocation remains.
