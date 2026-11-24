# Current drawer opening from the public camera

On the eight archived Kinova3 review trajectories, the new cue removed all 20 premature placement calls in the 248 replans from controls 45 through 195, with zero missed openings. It switched back to “open the top drawer” at the first sampled reclosure on both affected trajectories. The model was frozen on Panda data before this readback. The endpoint is cue correctness on the fixed BASE camera. [Archived cases and model digest](drawer-review-evaluation.json) · [Original review](../pr10-review-20261124-r1/drawer-state.json)

| Evaluation | Frames | Previous premature / missed | Current premature / missed | Previous / current mean Brier |
| --- | ---: | ---: | ---: | ---: |
| 10 independent Panda demonstrations | 403 | 8 / 0 | 0 / 0 | 0.01591 / 0.00000 |
| 4 independent Panda layouts with controlled drawer travel | 480 | 139 / 4 | 0 / 24 | 0.22600 / 0.03333 |
| Archived Kinova3, four scheduled controls per trajectory | 32 | 1 / 0 | 0 / 0 | 0.03029 / 0.00617 |
| Same trajectories, every replan from 45 to 195 | 248 | 20 / 0 | 0 / 0 | 0.06574 / 0.00081 |
| Same trajectories, all sampled controls from 0 to 300 | 488 | 20 / 0 | 0 / 0 | 0.03344 / 0.00041 |

The controlled Panda sweeps crossed from open to closed 16 times. The new cue recognized all 16 on the first reclosed frame. All 24 missed openings were at −0.1405 m, half a millimetre past the strict native boundary. The old cue advanced on 119 of 252 frames that were closed after an earlier opening. The new cue advanced on zero. Current minus previous mean Brier had a 95% paired whole-trajectory bootstrap interval of −0.02136 to −0.01045 on the demonstrations and −0.22714 to −0.15819 on the four sweep layouts. These intervals condition on the recorded trajectories and sweep design. [Per-case source evidence](drawer-source-evaluation.json)

On Kinova3 layout 31, tape 14, the first sampled reclosure was control 105. The old placement score was 0.962 and the new score was 0. At control 140 the displacement was −0.09849 m and the scores were 0.797 and 0. On layout 33, tape 14, the new cue switched back at control 100. The eight replays reproduced every archived 16 × 16 public tile and every archived drawer joint readback exactly. Layouts 36–39 were excluded throughout. Native task-success effects remain unmeasured here. [Frame readbacks](drawer-review-evaluation.json)

## Source calibration and split

The cue uses the fixed camera strip previously used to annotate the Panda demonstrations. In upright 128-pixel coordinates it searches rows 73–80, columns 15–87. On each visible scanline it interpolates the first dark-wood to bright-table crossing at gray 90. The median x coordinate, measured relative to the actual reset image, tracks drawer travel. The four previous camera tiles still enter the selector's progress features. [Original source annotator](../frozen-vla-phase-transfer-20261123-r1/annotate.py) · [Runtime implementation](signals.py)

The fit used original human Panda HDF5 demo IDs 20, 21, 24–35 and 40–43. Image index j matches state j+1, verified against all seven recorded arm joints. For every crossing of the native condition `wooden_cabinet_1_top_level < -0.14` m, the fit interpolated the visible displacement between the two bracketing observations. Each demo contributed its median crossing displacement, ranging from 18.8995 to 19.5922 reference pixels. The public score is the fraction of those 18 displacements that the current front exceeds. A strict majority issues placement. A tie or an unobservable reset/current edge issues opening. The fitter reads only the reset and crossing images from each fitting trajectory. [Fitted thresholds](drawer-calibration.json) · [Fit and score code](calibrate_drawer.py)

An initial tile screen used source demos 20–35 and 40–49 and source sweep layouts 20–23. For the subsequent development comparison, candidate fitting excluded eight human trajectories (22, 23, 44–49) and two controlled Panda layouts (22, 23) with matching pose-donor IDs. Those exclusions supplied the development observations. A single luminance tile fitted with demonstration and sweep labels missed 56 of 240 development sweep states. Forward selection with up to four tiles reduced that to 40. The source cabinet-edge calibration missed eight, all 1 mm from the boundary, with zero classification errors on the eight development demonstrations. The final fit uses the same 18-demo edge calibration. It requires no renders or optimizer.

The final source evaluation used human demos 0–9 at five-control intervals. Those demos contained no sampled reclosures. A separate controlled check used released Panda reset-bank layouts 24–27, seeds 42710324–42710327 and eight Panda arm/gripper postures per layout from held demos 0–3 in order. Each leg held the arm still while the drawer traversed 15 known positions. Alternating the direction provided repeated reclosures independent of arm motion. These are software-rendered pose interventions. Only RGB entered the cue. Demo and released layout numbers index different datasets. Across all fit/evaluation pairs, the smallest maximum separation of the four corresponding initial object positions was 9.34 mm for held demos and 71.92 mm for held released layouts. [Historical scene audit and data origin](../frozen-vla-phase-transfer-20261123-r1/LINEAGE.md)

## Baige reproduction

The integrated implementation and tests are at `85696bd8231b8115cc0d7959a1662dad46bb71b2`. The original source dataset revision is `f13aa24a3da8c43c7225569f28c562979fa0e35a`. The fitter verifies its full HDF5 SHA-256 (`1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8`). The native release is `e5500c690c1498eca1b7050350654964190cdf70`, the Panda reset-bank SHA-256 is `1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f`, and the historical CPU wrapper pins the arm environment at `a8a997e7d06bb131d3f9710204c58e106f0f0904`. The runtime uses the project's `setups/libero-meta-learning-20261117-r1/assets` and LIBERO config. The wooden cabinet XML there resolves into the pinned release and hashes to `eca251258c7ff6a05fa181a316c767508801812f4ea870921f3ce28fd82ea7af`.

The Baige run used CPU affinity 30,31, one BLAS thread, hidden CUDA and Mesa llvmpipe (LLVM 15.0.7). Python was 3.11.14 with NumPy 1.26.4, h5py 3.16.0, SciPy 1.15.3, robosuite 1.4.0 and MuJoCo 3.2.3. Git bundles supplied the detached scratch checkouts under `/mowenjie/fluffy-bassoon-2/worktrees/pr10-current-drawer.Iz8VEO`. Refitting the integrated implementation reproduced the shipped calibration byte for byte. All 51 selector tests passed there on Linux.

Run from a checkout containing these evidence files. The following Bash commands reproduce the fitted model, held source scores, archived review scores and tests.

```bash
set -eu
code=$(git rev-parse --show-toplevel)
s=$code/outputs/frozen-vla-selector-20261124-r1
root=/mowenjie/fluffy-bassoon-2
phase=$root/outputs/frozen-vla-phase-transfer-20261123-r1
cpu=$root/outputs/frozen-vla-reachability-20261123-r1/final-b1cf4be383c928001d5a8af2eb4ef190d043d632/cpu.sh
work=$(mktemp -d "$root/worktrees/pr10-current-drawer.XXXXXX")
mkdir "$work/tmp"
export REACH_CPUS=30,31 REACH_CACHE=$work/cache XDG_CACHE_HOME=$work/cache TMPDIR=$work/tmp
bash "$cpu" "$s/calibrate_drawer.py" fit "$phase/source.h5" "$work/model.json"
cmp "$work/model.json" "$s/drawer-calibration.json"
bash "$cpu" "$s/calibrate_drawer.py" score "$phase/source.h5" "$work/model.json" --report "$work/source.json"
cmp "$work/source.json" "$s/drawer-source-evaluation.json"

awk '/^```python$/{body=1;next} body && /^```$/{exit} body' "$s/drawer-evidence.md" > "$work/readback.py"
bash "$cpu" "$work/readback.py" "$s" "$work/review" "$phase"
cmp "$work/review/review.json" "$s/drawer-review-evaluation.json"

export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$code/src
taskset -c "$REACH_CPUS" "$root/setups/libero-meta-learning-20261117-r1/env/bin/python" -Bm pytest \
    -q -p no:cacheprovider --basetemp="$work/pytest" "$s"
```

The archived review requires full-resolution exposures. This readback uses the historical capture driver, verifies the locked source calibration digest, and saves public predictions before joining the archived labels.

```python
import json
import sys
from pathlib import Path
import numpy as np

selector, out, phase_run = map(Path, sys.argv[1:])
sys.path.insert(0, str(selector))
from signals import PHASE_DIR, drawer_front, drawer_probability
sys.path.insert(0, str(PHASE_DIR))
import capture
from common import save, sha, tiles
from reader import predict

model = selector / "drawer-calibration.json"
assert sha(model) == "49c42bbfc3be02a20e93f564ad4d8617e87db33c29c51396c3fb3d74860bf534"
thresholds = np.array(list(json.loads(model.read_bytes())["crossing_px_by_demo"].values()))
old = json.loads((PHASE_DIR / "evidence/model.json").read_bytes())
out.mkdir()
capture.tiles = lambda image: np.r_[tiles(image), np.float64(drawer_front(image))]
results = {}
for layout in range(30, 34):
    for donor in (14, 15):
        case = f"row{layout}-tape{donor}"
        capture.record(capture.NATIVE / f"libero_goal-3-{donor}-Kinova3.json",
                       layout, 42710300 + layout, False, out)
        data = np.load(out / f"public-{case}.npz")
        t, values = data["t"], data["tiles"]
        first = values[0, -1]
        p = np.array([drawer_probability(first, v, thresholds) for v in values[:, -1]])
        save(out / f"pred-{case}.json", dict(probability=p.tolist(), model_sha256=sha(model)))

        archived = phase_run / "crossed01"
        public = np.load(archived / f"public-{case}.npz")
        np.testing.assert_array_equal(t, public["t"])
        np.testing.assert_array_equal(values[:, :-1], public["tiles"])
        truth = json.loads((archived / f"truth-{case}.json").read_bytes())["q"]
        np.testing.assert_array_equal(truth, json.loads((out / f"truth-{case}.json").read_bytes())["q"])
        q = np.array(truth)[t]
        y = q < -0.14
        old_p = predict(t, public["tiles"] - public["tiles"][0], old)["instant"]
        cases = {}
        for key, mask in (("scheduled", np.isin(t, [45, 60, 75, 90])),
                          ("replans", (t >= 45) & (t <= 195)), ("all", t >= 0)):
            stats = {}
            for name, prob, advance in (("previous", old_p, old_p >= .5), ("current", p, p > .5)):
                stats[name] = dict(premature=int(np.sum(mask & advance & ~y)),
                                   missed=int(np.sum(mask & ~advance & y)),
                                   brier=float(np.mean((prob[mask] - y[mask])**2)))
            cases[key] = dict(frames=int(mask.sum()), **stats)
        wrong = (p > .5) != y
        events = np.flatnonzero(np.r_[False, y[:-1] & ~y[1:]])
        def frame(i):
            return dict(t=int(t[i]), q_m=float(q[i]), current=float(p[i]), previous=float(old_p[i]))
        results[case] = dict(windows=cases, first_reclosed_frames=[frame(i) for i in events],
                             current_errors=[frame(i) for i in np.flatnonzero(wrong)],
                             control_140=frame(list(t).index(140)),
                             public_sha256=sha(archived / f"public-{case}.npz"))
        print(case, cases["replans"], flush=True)
save(out / "review.json", dict(model_sha256=sha(model), cases=results))
```
