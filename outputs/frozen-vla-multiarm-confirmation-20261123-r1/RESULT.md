# Native arm screen qualification in simulation

**Both arms passed CPU qualification for parent submission.** The eight layout-14 task identities matched across arms on the released scene, BDDL, reset plan, prompt, native action interface, horizon and flow-noise bank. Their compiled models have distinct arm XMLs. Joint-position, joint-velocity and per-control joint-tape checks used the joint names from each compiled model. UR5e presented six physical joints and IIWA seven. Both used the common seven-component 20 Hz OSC pose and gripper command. [Readback](evidence/cpu-readback.json) · [Full reset, action and native reward receipts](evidence/cpu-receipts.tar.gz)

| Simulated arm | Native raw success on historical layout 13 | Layout-14 identities qualified | CPU resets | Scripted native controls |
| --- | ---: | ---: | ---: | ---: |
| IIWA | 6/8 | 8/8 | 12 | 66 |
| UR5e | 8/8 | 8/8 | 12 | 66 |

On each arm, libero_10/8 and libero_goal/6 exercised raw, planar and full wiring for ten controls each. The other six identities exercised one native control each. Both paired action chunks matched the registered translation matrix and clipping operation, and rotation and gripper commands were identical across scripted methods. The two camera inputs, initial physics, noise and opening proposal matched within each paired reset. Across arms, each policy receives its own simulator observations. All 16 calibration tapes matched their locked byte hashes. Each provided one 15-control trajectory on layout 11 with a distinct query reset. [Qualification code](qualify.py) · [Pair checks](screen.py)

Historical IIWA failures were libero_10/8 and libero_goal/3. The fresh screen retains both alongside six historically working IIWA inputs and all eight historically working UR5e inputs. The native task report records candidate rescues among actual reference failures, adverse effects among actual reference successes, and the full-vs-planar contrast. It also records layout-13 to layout-14 raw occupancy, native reward occupancy by control, missing cells and uncertainty. CPU tests exercised empty rescue and harm denominators, simultaneous rescue and harm, and a missing panel. The fixed CPU action prefixes establish simulator wiring. Native success, rescue and harm on layout 14 remain unmeasured by this qualification. [Paired report](report.py) · [Statistics checks](evidence/statistics-wiring.json)

| Revision | Purpose |
| --- | --- |
| `06013c21cda1bc6a2a78eb2a92239dd7006167f0` | Design freeze before accessing new Kinova outcomes |
| `a8a997e7d06bb131d3f9710204c58e106f0f0904` | Locked native runtime and frozen-policy runner |
| `c8e24a09abd53e34734d6f51a82548c8a961da78` | Baige CPU qualifier, cores 38 and 39 |
| `cb246fffcec1d620603a7da841d29199568fcfa0` | Deployed GPU entry point and native report |
| `d9287fb802b5ff31b21581671d3d58c8353927a8` | Independent CPU receipt readback |

Baige CPU qualification exited zero. The GPU launch revision uses the same CPU-tested adapters, runner and pairing checks. Its terminal check accepts a simulator-reported early termination and requires a truncation to reach the released horizon. The Linux readback checked all 16 receipts and exercised the pinned reporting code. GPU consumption by this workstream during qualification was 0 GPU-hours. The screen fixes layout 14 only, with 24 native episodes and a maximum of one GPU-hour per arm. The two commands and runtime bindings are in [the parent handoff](READY.md).

To reproduce the CPU qualification on Baige with a fresh output name:

```bash
bash /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-multiarm-confirmation-20261123-r1/src-c8e24a09abd53e34734d6f51a82548c8a961da78/outputs/frozen-vla-multiarm-confirmation-20261123-r1/cpu.sh c8e24a09abd53e34734d6f51a82548c8a961da78 cpu-repeat01
```
