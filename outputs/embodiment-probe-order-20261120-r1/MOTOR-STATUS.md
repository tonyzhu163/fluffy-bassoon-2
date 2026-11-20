# Existing PaliGemma source jobs

Read-only Baige observations from November 20, 2026, 15:52–15:54 Shanghai. The three generated demonstration collections each exited 0, and each `banks/campaign.json` recorded `complete`. Counts below are retained NPZ files under the four bank directories. The runs share revision `17338c926b24a8eb0ceb985b1a0db90f745c15ab`. The project PFS root for these receipts is `/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-dualarm-20261120-r2`.

| Directory | Job ID in launch receipt | bank0 | bank1 | bank2 | bank3 |
| --- | --- | ---: | ---: | ---: | ---: |
| gendemos01 | `job-isezj1jp5rg6` | 37 | 44 | 40 | 39 |
| gendemos02 | `job-bt2z25374394` | 55 | 54 | 56 | 56 |
| gendemos03 | `job-1da0x4xol75p` | 56 | 60 | 58 | 47 |

The broader source fit `broadfit01` (`job-l5dd256gvivo`) exited 0 and retained candidates at steps 1,600, 3,200 and 6,400. The recovery fit `recoverfit01` (`job-4kvvitcsw1nh`) exited 0 and retained candidates at steps 600, 1,200 and 2,400. The earlier `recover01` collection (`job-lf7wfg7kuxrh`) recorded exit 1. The corrected-panel scorer `hostrecover01score` (`job-1w5htplffp89`) and stride development run `stride01` (`job-ybo867hh45g6`) recorded exit 0.

The scheduler list also showed successful waypoint-50 (`job-qhlx3223np0a`), categorical local-50 (`job-3bffdcmp7clx`), phase fit (`job-cdoxszpm27hb`) and numerical regression-50 (`job-9zpwf7aapj3u`) jobs. The source workstream maintains the task-success comparisons. [Source execution record](../libero-paligemma-dualarm-20261120-r2/EXECUTION.md)

## Native source development scores already present on the project mount

At 15:55 the finished reports at `<directory>/dev/scores/<checkpoint SHA>/stride-5/report.json` each contained 50 episodes per arm and `passed=false`. The 5,200-step waypoint checkpoint already has a completed development evaluation. The readback resolves the conditional waypoint launch in [the source audit](MOTOR.md).

| Directory | Update | Panda | Sawyer | SHA-256 of report on PFS |
| --- | ---: | ---: | ---: | --- |
| large-panel-hostway01 | 5,200 | 1/50 | 1/50 | `ad3a08052ed7e410e9c98717c8495af05cc6fa4ce5c61382105e7511c0d3aa54` |
| large-panel-hostrecover01 | 2,400 | 17/50 | 15/50 | `76737313e15bbfea22cea1745859777ffdf45eaba4f174b9077435ea8ce19a73` |
| large-panel-hostcat01 | 6,400 | 14/50 | 15/50 | `9e0dd7754ba86d09e7a64447ee9c580be60f85164eff430103740dd3403bed46` |
| large-panel-v3 | 6,400 | 34/50 | 20/50 | `7e2122efbd976638680c8215c3830eeb974179c5944f5185b38771988a3d464a` |

The four bank indices in each collection gave the following admitted-layout and accepted-demonstration counts. The accepted counts match the NPZ file totals above.

| Collection | Admitted layouts | Panda accepted | Sawyer accepted |
| --- | ---: | ---: | ---: |
| gendemos01 | 96 | 66 | 94 |
| gendemos02 | 128 | 93 | 128 |
| gendemos03 | 128 | 98 | 123 |

On the four paired starts in `stride01/dev/report.json`, strides 1 and 2 each scored 3/4 Panda and 0/4 Sawyer. Holding five actions scored 2/4 Panda and 1/4 Sawyer. Against stride 5, stride 1 had two wins and one loss on Panda, and stride 2 had one win and zero losses. Both shorter strides lost the same Sawyer success at pose 35 and rescued none.
