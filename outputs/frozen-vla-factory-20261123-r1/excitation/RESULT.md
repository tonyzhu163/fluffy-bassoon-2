# Two directions identify a wider planar interface on UR5e

Spending the three bursts on the first direction, its perpendicular, then the first direction again reduced mean relative full-map error from **0.560 to 0.270** against passive-template directions. The paired reduction was **0.290** (95% task-bootstrap interval **0.204–0.363**). Against repeating one direction the reduction was **0.669** (0.646–0.694). These are measurements from 384 fresh, headless LIBERO-Pro2.0 BASE **support** simulations. They test an adapter's identification of the commanded XY transformation. No corrected VLA query or task-success evaluation ran.

## Fresh simulated interventions

The public directions came from 32 archived frozen π₀ or π₀.₅ passive command templates on UR5e. For each task and motor we took a low- and high-spread template before scoring. We constructed all three schedules, replayed them open loop in the same settled native scene, and measured grip-site displacement with trusted FK. Even the passive-template condition uses standardized XY magnitude, zeroes the other motion axes, and holds the jaw open. The original policy recordings appear separately below.

Each triplet shared an exact process fork of one layout-11 reset, a hidden 2×2 map, three command norms, a 0.75 s duration and the same added displacements. The test crossed both determinant signs at singular values 1.45 and 0.70, with one pair of orientations drawn per task. Physical command components stayed within the controller limits. Map-induced differences in travel are part of the intervention. Noise-free below means the native wrist measurement without added jitter. For the noise stress, a common four-endpoint Gaussian draw induces correlated displacement errors with per-axis RMS 2.52 mm in expectation. The scale comes from source-model residuals, so it includes controller mismatch. The draw is shared within each triplet and varies between triplets.

| Schedule or subset | One direction | Passive directions | Two directions | Two minus passive, paired 95% interval |
|---|---:|---:|---:|---:|
| All, with added noise | 0.940 | 0.560 | 0.270 | −0.290 [−0.363, −0.204] |
| Weak pulses, norm 0.155 | 0.948 | 0.697 | 0.411 | −0.286 [−0.384, −0.193] |
| Strong pulses, norm 0.620 | 0.931 | 0.424 | 0.129 | −0.294 [−0.375, −0.197] |
| Low-spread passive template | 0.938 | 0.648 | 0.266 | −0.382 [−0.454, −0.302] |
| Higher-spread passive template | 0.941 | 0.472 | 0.275 | −0.198 [−0.295, −0.097] |
| π₀ command templates | 0.928 | 0.510 | 0.269 | −0.241 [−0.333, −0.134] |
| π₀.₅ command templates | 0.951 | 0.610 | 0.271 | −0.339 [−0.432, −0.241] |
| All, noise-free | 0.935 | 0.477 | 0.095 | −0.382 [−0.460, −0.275] |

Entries are mean relative forward errors `||A_hat − A||_F / ||A||_F` for the full ridge fit (lower is better). Each of eight tasks supplied 16 paired triplets. The percentile intervals resample the eight whole tasks, keeping both motors, both map signs and all repeated starts together. They describe sensitivity to these tasks at the fixed interface draws and jitter seed. There is one support start per task.

The command singular-value ratio was zero on one direction and 0.707 on two. Median passive ratios were 0.081 and 0.146 for the low- and higher-spread templates. Under matched added noise the scaled-rotation fit on the same two directions had error 0.740 versus 0.270 for full ridge. The latter improved on passive full ridge by 0.152 [0.097, 0.203] on positive determinants and 0.428 [0.281, 0.563] on reflections. The unadapted identity averaged 1.472 across these maps.

## Original recorded supports

On all 192 original frozen-policy supports, the archived target transforms are scaled rotations. Mean relative error was **0.080 for the scaled-rotation fit** and **0.262 for full ridge** (paired full minus rotation +0.182 [0.147, 0.222]). The median command singular-value ratio was 0.093. The four-parameter fit adds variance on these recorded narrow-family interfaces. At both recorded motors the constrained fit won. The previously reported full-map collapse under passive support used a different motion reader and a residual-after-inversion metric. [Earlier wider-family screen](../../symmeta-wider-family-20261123-r1/RESULT.md)

The reader fits two nominal Panda coefficients on four 15-control pulse tapes from Goal tasks 5 and 7. An effective chunk exposure is 0.04729 times the current command plus 0.01290 times the previous one, in metres. The ridge prior is the scaled rotation fitted from the same three observable displacements. Its regularization scale is the squared source residual RMS. Frozen motor weights, task labels, condition identities and reward are absent from estimation. The eight test task IDs are disjoint from calibration. Recorded yaw/gain enters only at scoring. Injected matrices enter the prospective actuator wrapper and the scorer. [Source NPZ protocol](../../libero-xemb-joint-sysid-20261119-r1/PLAN.md) · [UR5e π₀.₅ archive and FK qualification](../../symmeta-camera-command-20261123-r1/RESULT.md) · [UR5e π₀ qualification](../../symmeta-fk-crossarm-20261123-r1/RESULT.md)

## Inverting a weak estimate

At amplitude 0.155, two-direction ridge had median residual-after-inversion error 0.446 but a mean of **2.039** and 95th percentile **9.493** with added noise. The one-direction and passive means were 0.923 and 1.513. At 0.620, two-direction mean residual fell to **0.158** versus 0.908 and 2.564, with 95th percentile 0.335. Better forward identification can still produce extreme inverse commands at low signal. Residual here is `||A pinv(A_hat) − I||_F / sqrt(2)` prior to command clipping. It measures how the estimated correction acts on two independent command axes.

For a frozen motor facing a wider interface, test a perpendicular second burst plus a full-map adapter when its motion is measurable. Include a bounded or regularized inverse in the next paired retry. Retain a scaled-rotation baseline on yaw/gain panels. This result depends on a trusted world reference, one arm, fixed hidden singular values and short open-loop XY motion. It leaves visual-only reading, lost task time, contact, closed-loop policy responses and grasp success open.

## Receipt and reproduction

Original repository revision `757a44cd4374833cad790852df8b87ddf412e638`. One local Linux CPU collector ran seed 20261123, with rendering disabled, sequentially forking one rollout at a time. All 384 support rollouts completed 15 controls for 5,760 counted controls plus setup/settling, using **0 GPUh**. Every branch checked its parent's reset-state digest. The maximum start-joint difference against the archived UR5e reference was 5.87×10⁻⁸ rad. Fresh matched openings are established within the experiment. The original archived runs have different whole-simulator fingerprints. The compact support receipt includes eight reset hashes, task seeds, stock arm and gripper hashes, source tape IDs, all commands, FK displacements and hidden scoring matrices. [Support receipt](supports.json.gz) · [Metrics including inverse tails, controls and per-task differences](metrics.json) · [Plan](PLAN.md)

A separate full collection and score rerun exited zero and reproduced both archived files byte for byte (support SHA-256 `b2b715bd25d099832dfcf40fde59d69c94d545615d29591ecb5b87f6c3881afa`, metrics SHA-256 `13227d50a0a7cf484080e73eec73fb273c595ee68e8e64d4c7f080ec0e75bb18`). The rerun repeated the same 5,760 controls and contributes no additional independent cases.

Rescore with NumPy from the repository root:

```bash
lane=outputs/frozen-vla-factory-20261123-r1/excitation
python "$lane/score.py" --out /tmp/excitation-metrics.json
```

To repeat the physical simulation on Linux, use the released LIBERO-Pro2.0 checkout and an isolated CPython 3.11 environment. The BASE scenes used here rely on assets present in the regular Git checkout. No policy checkpoint is needed for command-template replay.

```bash
lane=outputs/frozen-vla-factory-20261123-r1/excitation
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0.git /tmp/excitation-libero
git -C /tmp/excitation-libero checkout e5500c690c1498eca1b7050350654964190cdf70
uv venv --python 3.11 /tmp/excitation-runtime
py=/tmp/excitation-runtime/bin/python
uv pip install --python "$py" 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$py" 'robosuite==1.4.0' 'mujoco==3.2.3' 'numpy==1.26.4' \
  'numba==0.61.2' 'scipy==1.15.3' 'termcolor==2.5.0' 'bddl==1.0.1' 'easydict==1.9' \
  'pyyaml==6.0.2' 'h5py==3.12.1' 'gym==0.26.2' 'future==1.0.0' 'requests==2.32.4' \
  'cloudpickle==3.1.1' 'imageio==2.37.0' 'pydantic==1.10.22' 'httpx==0.24.1' \
  'msgpack==1.0.8' 'transforms3d==0.4.2' 'matplotlib==3.9.4'
bash "$lane/reproduce.sh" "$py" /tmp/excitation-libero /tmp/excitation-repeat
```
