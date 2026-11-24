# Damping the command inverse on weak supports

On 64 weak two-direction supports, uncertainty-weighted damping reduced mean XY error on replayed frozen π₀.₅ commands from 0.308 to 0.184. The paired change was −0.125 in normalized command units (95% whole-task bootstrap interval −0.177 to −0.077). Damping also reduced the large residual of the unclipped inverse. The same correction increased error on the 192 original yaw/gain supports. [All schedules, tail errors and paired intervals](metrics.json)

| Support panel | Plain full inverse | Directional damping | Isotropic damping | Scaled rotation |
| --- | ---: | ---: | ---: | ---: |
| Two weak directions, 64 supports | 0.308 | 0.184 | 0.189 | 0.449 |
| Two strong directions, 64 supports | 0.067 | 0.065 | 0.066 | 0.446 |
| Recorded yaw/gain, 192 supports | 0.104 | 0.136 | 0.147 | 0.039 |

The endpoint is mean XY distance after clipping the issued command, applying the held-out interface and clipping the resulting physical command. Eight archived development tapes contribute equally per support. Truth enters only the scoring step. The strong-probe paired change was −0.0018 [−0.0040, +0.0002]. On the recorded supports it was +0.0314 [+0.0199, +0.0442]. The close isotropic result on weak probes leaves the need for direction-specific damping unresolved. [Support interventions](../excitation/RESULT.md) · [Command provenance](../native/RESULT.md)

Use scaled rotation as the baseline for yaw/gain. For reflected, anisotropic command maps with weak excitation, include a damped full inverse in the next paired rollout. Each task here has one simulated support reset, and the action tapes stay fixed under all corrections. Online replanning, actuator interactions and task reward remain unmeasured.

## Reproduction

On Linux CPU with NumPy 2.5.4, from the repository root run the command below. Support and command bank inputs are pinned by the excitation and native lanes at repository baseline `757a44cd4374833cad790852df8b87ddf412e638`. All 384 simulated supports reuse the excitation lane's endpoint jitter at seed 20261125. The interval resamples eight tasks 20,000 times at seed 20261123.

```sh
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -B outputs/frozen-vla-factory-20261123-r1/inverse/screen.py
```
