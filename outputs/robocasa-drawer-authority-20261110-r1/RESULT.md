# Five-control drawer offset on two reproducible openings

The frozen Human300 π0.5 policy opened the drawer within 600 controls on two of eight prespecified development seeds. Both openings survived a 4 cm lateral offset in either direction without any correction. The rescue fraction is undefined because every uncorrected branch succeeded. The six nominal failures reached the horizon and were ineligible under the [protocol](PLAN.md).

| Seed | Nominal opening at control | Last offset control | First nominal two-finger grasp | Opening with −4 cm, no correction | Opening with +4 cm, no correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 982101 | 273 | 175 | 211 | 263 | 277 |
| 982103 | 260 | 130 | 193 | 220 | 257 |

Both nominal command replays matched the original physical trajectory through terminal success. All 20 branches matched their scene's simulator prefix and decision observation. Nineteen opened the drawer. The only timeout came on seed 982103 when a −2 cm residual reinforced the −4 cm offset. The opposite 4 cm residual opened the drawer in all four scene–sign pairs, as did zero residual. [Measurements](SUMMARY.json) · [Per-episode records](events.jsonl).

The offset ended 36 and 63 controls before the first nominal two-finger grasp. Retest these two scenes with the imposed offset active through gripper closure and compare zero correction with the known inverse at the same decision. A repeatable failure and recovery would justify expanding the development panel.

Baige Pro6K job `job-g3i91figsii6` ran source `e8dec2e61c08169b7e7db14363bc83563be3629e` on one GPU. The worker exited successfully after 1,244 seconds (0.346 GPU-hours), completing 30 episodes and 11,108 controls including screening and replay. [Source identity](identity.txt) · [GPU probe](gpu.json) · [Worker clock](exit.json) · [Scheduler accounting](scheduler-close.png).

After worker exit we requested release. Baige showed the drawer job stopping at 23m11s of task time (0.386 GPU-hours). Including the phone-stand replay, the two workers used 0.407 GPU-hours in total. The two scheduler receipts showed 0.522 GPU-hours of task time at closeout. [Phone-stand accounting](../robotwin-e1-replay-20261110-r1/RESULT.md).
