# Lower finger friction cost default two successes. True material added zero.

On eight held-out LIBERO development tasks, fixed default earned native return **8/8 under ordinary friction and 6/8 under reduced friction**. Fixed pacing earned **6/8 under each**. The best rule choosing a program from true material tied fixed default at **14/16**. All 32 branch outcomes were measured in native BASE physics on query reset 1, repeat 0. The endpoint is open-loop success within each archived frozen π0.5 action tape. [Registered plan](PLAN.md) · [Verified readback](cpu01/readback.json)

| Program on ordinary material | Program on reduced material | Ordinary /8 | Reduced /8 | Total /16 |
| --- | --- | ---: | ---: | ---: |
| Default | Default | 8 | 6 | 14 |
| Default | Paced | 8 | 6 | 14 |
| Paced | Default | 6 | 6 | 12 |
| Paced | Paced | 6 | 6 | 12 |

Switching to pacing under reduced friction recovers **Object 7**, the milk placement, and loses **Object 0**, the soup placement. On **LIBERO-10 task 0**, the two-can placement, both programs miss under reduced friction. Choosing a single program per task achieves **15/16** by pacing Object 7 and using default elsewhere. Allowing that task-specific choice to depend on material also achieves **15/16**. None of the eight tasks requires opposite programs across its two materials. These are within-panel selection ceilings on one start per task. [All program and material assignments](cpu01/score.json)

## Tape lengths, endpoints and native reward

Each outcome cell gives the ordinary and reduced stop controls, in that order. A number alone means native success and return 1 at that control. **H** means horizon truncation with return 0. **E** means the recorded tape ended with return 0 while the episode still had control budget. Native reward equaled the binary success predicate on all 7,322 controls, with zero reward before first success. [Per-branch measurements](cpu01/branches.jsonl) · [Source seeds, tapes and prompts](cpu01/sources.json)

| Task | Horizon | Default tape length | Default stops, ordinary / reduced | Paced tape length | Paced stops, ordinary / reduced |
| --- | ---: | ---: | --- | ---: | --- |
| LIBERO-10 0 · Two cans into basket | 520 | 281 | 281 / **281 E** | 520 | 520 H / 520 H |
| LIBERO-10 3 · Bowl into drawer and close | 520 | 206 | 206 / 206 | 507 | 507 / 506 |
| Goal 0 · Open middle drawer | 300 | 106 | 106 / 106 | 267 | 267 / 267 |
| Goal 1 · Bowl onto stove | 300 | 82 | 82 / 82 | 267 | 267 / 267 |
| Object 0 · Soup into basket | 280 | 133 | 133 / 133 | 280 | 280 H / 280 H |
| Object 7 · Milk into basket | 280 | 127 | 127 / **127 E** | 272 | 272 / 272 |
| Spatial 1 · Bowl beside ramekin onto plate | 220 | 95 | 95 / 95 | 200 | 200 / 197 |
| Spatial 7 · Bowl on stove onto plate | 220 | 118 | 118 / 118 | 202 | 202 / 202 |

Fourteen of the 16 source tapes ended on an ordinary success before the horizon. Both newly lost default successes exhausted such tapes, with **239 controls** remaining on LIBERO-10 task 0 and **153** on Object 7. Their outcomes under longer programs or live replanning remain unmeasured. Two reduced pacing branches reached success one and three controls earlier than their archives. Across both materials there were 26 native successes, four horizon truncations and two tape-end misses. No commands were appended. [Per-control trace inventory](checksums.sha256)

## Physical and reproduction checks

The requested 0.05 sliding coefficient was applied after common settling to the same four finger collision geoms as in the earlier gate. **All 16 program pairs** had a common active finger–object or finger–fixture geom pair with changed realized friction. On matched pairs the measured sliding coefficient fell from **2.00 or 1.00 to 0.95**. Object 7 showed both drops on the milk carton. Goal 0 showed the 1.00 to 0.95 drop on the drawer fixture. The contact logger sampled every native integration step. [Changed coefficients by geom pair](cpu01/pairs.json) · [Original material gate](../libero-xemb-contact-20261119-r1/RESULT.md)

All four branches for each task matched in reset physics fingerprint, full MuJoCo integration-state hash, body and camera poses, and the five public robot fields. Every branch matched the archived physics fingerprint and public robot fields. The 16 ordinary replays reproduced all source outcomes, control counts, reward sequences and **all 3,663 post-action wrist positions exactly**. In the 16 material pairs the five public fields matched at all **1,401** sampled control boundaries from reset through the last complete precontact control. Issued commands, rewards and termination flags matched over those prefixes. [Trace readback](cpu01/readback.json)

Both 256 × 256 cameras were physically sampled at frame zero on all 32 branches. Agentview and wrist RGB hashes matched within **16/16 material pairs** for each camera. Rendering preserved the integration, body and camera state hashes on every reset. The software-rendered hashes differed from the archived GPU hashes for both cameras on every start. Image equality is established between the fresh material branches at frame zero. Later precontact comparisons cover proprioception, actions, rewards and episode flags. Per-step cameras were disabled because replay consumes recorded actions. [Frame hashes, renderer and state hashes](cpu01/branches.jsonl)

Baige development CPU allocation `aibox-rffe453e3884-5f96c78b88-6ssrn` ran one simulator process at detached registration `a6d6cb8d208bf8df744986e03c802a89343fe8d3`, release `e5500c69`, archived frozen motor queries from `paired01` at `c45aa965`. Native execution ran 20:11:12–20:15:30 Shanghai on November 19, 2026, **258 seconds**, exit 0, **0 GPU-hours**. The project mount had 477 GiB available and passed a write/read check. The first bundle fetch refused a checked-out ref before simulation. Fetching the verified bundle into a fresh ref in the private clone supplied the detached pin. All run writes stayed under the owned directory. The readback scored cached outcomes after exactly 32 replays. Only the eight development tasks were executed. [Preflight](preflight.txt) · [Bundle deployment](deploy.log) · [Launch](cpu01/launch.txt) · [Exit](cpu01/exit.txt)
