# Data, splits and reproducibility

## Original source

The source was the 1,017,886,342-byte original LIBERO HDF5 for “open the top drawer and put the bowl inside,” revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, SHA-256 `1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8`. The source loader checks the full digest. All 50 original human Panda demonstration groups were inventoried. IDs 20–49 supplied the fit, 0–9 supplied the complete-episode source holdout and 10–19 were quarantined. Demonstrations 0 and 1 had already appeared as privileged pose donors in the geometry work. They were excluded from fitting. Neither Panda policy replays nor earlier Paddle data entered training. [Pinned original HDF5](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5) · [Source IDs and audit](evidence/split.json)

The released goal-3 Panda reset bank has SHA-256 `1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f`. Compare a pair of starts by taking the largest Euclidean separation among their corresponding initial bowl, cheese, wine-bottle and plate positions. To be treated as a duplicate, all four must coincide within the inherited 1 mm native reset scale. Across the 30 fit demos, the 10 held demos and registered target rows 14–17, 30–33 and 36–39, the smallest fit-to-holdout-or-target separation was 9.34 mm. Between distinct registered target rows the minimum was 11.47 mm. All four crossed layouts had distinct settled reset hashes, with exact equality between their two command branches. The task, fixed camera convention and objects are shared across the split. The frozen VLA's prior exposure to LIBERO is unverified. [Full input and reset accounting](evidence/lineage.json)

The source encoder consumed raw stored `obs/agentview_rgb` only. To annotate the 30 fitting demos, a source-only inspection located the cabinet front against the table in upright 128 × 128 images. The labeler measured its edge across eight horizontal lines (rows 73–80), relative to its initial and final visible positions. The switch was the first of two adjacent sampled frames showing at least seven eighths of that apparent travel. The fraction comes from the published cabinet's 14 cm open threshold and nominal 16 cm travel. Perspective and occlusion make this a weak image label. Individual physical joint values were excluded from all fitting, annotation and control calibration. The annotator uses the end image only offline. The evaluated reader has no access to a goal image, elapsed time, episode length, actions, embodiment identifier, task state, object coordinates, segmentation, wrist-camera stream, rewards or future observations. [Source annotation audit](evidence/annotation-audit.json) · [Frozen source model](evidence/model.json)

The historic dataset export skips five initial actions and stores observations after an action alongside pre-action state vectors. The elapsed source clock starts at the first available public image. Target time zero is the released settled reset. For each held demo the recorded seven public joint positions at image index (j) matched state (j+1) exactly (zero joint RMSE across the trajectory); same-index RMSE ranged from 0.0134 to 0.0167 rad. Source scoring used that observed offset and omitted the final unmatched image. Target scoring latched the strict native condition `q < -0.14` over every control endpoint and joined it to reset and every fifth camera observation. Public decisions are 0.25 s apart. Physical labels resolve 0.05 s control endpoints. Movement between those endpoints is unobserved. [Per-demo alignment and metrics](evidence/source.json) · [Source scorer](source.py)

## Target recordings

| Population | Parent data and identity | Unit of evaluation |
| --- | --- | --- |
| Known incidents | Native π₀.₅ job `job-xaugtearcacf`, commit `a8a997e7d06bb131d3f9710204c58e106f0f0904`, untreated Kinova3 goal 3 rows 14/15, seeds 41810314/41810315 | Two previously studied 300-control failures. CPU reenactments supply public images and local physical readbacks. |
| Fresh crossed tapes | Both original command arrays applied unchanged to every new native Kinova3 reset on rows 30–33, seeds 42710330–42710333 | Four layouts, two command sequences per layout. Whole-layout bootstrap with fixed command donors. All eight timed out at 300 controls. |
| Independent live raw starts | Feedback job `job-nms6mr8shrgx`, commit `522ccf30d18f96f4ad49104c9c862096a9db1652`. Expected `outputs/frozen-vla-tracking-feedback-20261123-r1/gpu01/libero_goal-3-{16,17}-raw.json` | Preregistered independent frozen-policy draws. At the 19:27 Shanghai Baige readback on November 23 neither raw tape existed. |
| Future intervention | Goal 3 only, Kinova3 BASE rows 36–39 at seeds 42710336–42710339 | Reserved, unexecuted by this study. Compare all three live branches on the same reset and noise. |

The native tape SHA-256 values are `5a5bd5bd03d521468bf32afee52bc1ea653b64fec6b8baaa53dd59aad9f75bec` and `b817b88f7664c1c67b7425e112e675d2b9ebc31c210760fa8b46ce6b30155894`. The collector uses the native Kinova3 model, physical two-finger gripper, world-frame OSC controller and official native task reward. It reads the actual public agent camera before the next decision. Its raw 256 × 256 image is pooled to the same normalized 16 × 16 grid as the 128 × 128 source image. The collector skips unused renders while replaying fixed actions. The full native IPC qualification rendered both ordinary streams and reproduced the expected public trigger times. No image of a second robot, mask, object state, future trajectory or intervention identifier enters a feature. [Input digests, reset hashes and reward outcomes](evidence/lineage.json) · [Native benchmark](../frozen-vla-native-transfer-20261123-r1/RESULT.md)

The four selected tiles have zero-based row/column coordinates (6,6), (9,6), (8,8) and (11,9) in raw image storage. The signed brightness difference from reset in each tile is standardized on fit demos. Its logistic coefficients are −2.118, −1.353, −1.269 and −0.190 with intercept 1.767. The frozen weights have SHA-256 `ca8906c86514d8917c1743726929c1e70aaf1f45e07de8892574c58a9ddf7008`. The scorer saved each complete prefix-checked probability trace before joining it to physical labels. Different features may track the visible hand, drawer or both. Transferring across arm and matched time does not identify the pixels' causal interpretation. [Model and timing probabilities](evidence/model.json) · [Known predictions](evidence/known-predictions.json) · [Crossed predictions](evidence/crossed-predictions.json)

## Relation to earlier work

The release experiment compared complete practice histories for choosing a retry instruction. On 256 trigger-matched assignments its clock and wrist readers achieved 182 and 181 native successes. The Paddle implementation specified learned phase and pose readouts for a successful plate-pushing teacher on source bodies. Its report documents no completed phase fit or native evaluation. Here the target is a native physical event in an ongoing drawer episode on a held arm. Earlier geometry work supplied the two Kinova incidents. The new starts change the scene under their identical recorded actions. VIP's goal-conditioned visual values and the factory's review of steering interfaces motivate separating evidence of progress from evidence of improved control. This study estimates a single task event under a fixed view. [Release timing](../libero-release-phase-20261119-r1/RESULT.md) · [Paddle phase protocol](../libero-paligemma-dualarm-20261120-r2/PHASE.md) · [Geometry](../frozen-vla-reachability-20261123-r1/RESULT.md) · [VIP](https://arxiv.org/abs/2210.00030) · [Factory review](../../docs/literature/vla-steering-review.md)

## Reproduction on Baige CPU

The run root is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-phase-transfer-20261123-r1`. The native run uses LIBERO-Pro2.0 release `e5500c690c1498eca1b7050350654964190cdf70`, robosuite 1.4.0 and MuJoCo 3.2.3. The existing qualified project Python environment supplied NumPy, SciPy, PyTorch for reset-bank loading, h5py and Pillow. CUDA was hidden. Mesa llvmpipe ran under CPU affinity 30,31. The project filesystem had 232 GB free and passed a write check before setup. No quota CLI was installed. Source preparation used commit `b313dd72`, native capture `39f08dc1`, annotations, fitting and scoring `c8c7e5be`, tape prompt qualification `c3550bea`, native IPC qualification `7a884cf0` and the final lineage readback `374130f8`. Each Baige checkout was detached at its verified Git revision. Original logs and full CPU control tapes are retained in the run root. Bootstrap seed was 20261123 with 20,000 whole-start resamples. Fitting was deterministic.

For a new reproduction set `c` to this run's directory inside the verified checkout and `r` to a fresh Baige results directory. Place the original HDF5 at `$r/source.h5`. The example uses Bash. It writes predictions before opening score labels.

```bash
set -eu
mkdir "$r/evidence01"
bash "$c/cpu.sh" source.py prepare "$r/source.h5" "$r/source01"
bash "$c/cpu.sh" annotate.py "$r/source.h5" "$r/source01"
bash "$c/cpu.sh" reader.py "$r/source01" "$r/source01/annotations.json" "$r/model.json"
bash "$c/cpu.sh" capture.py known "$r/known01"
bash "$c/cpu.sh" capture.py crossed "$r/crossed01"
bash "$c/cpu.sh" evaluate.py predict "$r/model.json" "$r/source01" "$r/evidence01/source-predictions.json" {0..9}
bash "$c/cpu.sh" evaluate.py predict "$r/model.json" "$r/known01" "$r/evidence01/known-predictions.json" row14-tape14 row15-tape15
bash "$c/cpu.sh" evaluate.py predict "$r/model.json" "$r/crossed01" "$r/evidence01/crossed-predictions.json" row{30..33}-tape{14,15}
bash "$c/cpu.sh" source.py score "$r/source.h5" "$r/source01"
for group in source known crossed; do
    bash "$c/cpu.sh" evaluate.py score "$r/evidence01/$group-predictions.json" "$r/${group}01" "$r/evidence01/$group"
done
bash "$c/cpu.sh" qualify_prompt.py "$r/model.json" "$r/source.h5" "$r/evidence01/source-predictions.json" "$r/evidence01/prompt-qualification.json"
bash "$c/cpu.sh" qualify_native.py "$r/model.json" "$r/evidence01/native-prompt.json"
bash "$c/cpu.sh" lineage.py "$r" known crossed
```
