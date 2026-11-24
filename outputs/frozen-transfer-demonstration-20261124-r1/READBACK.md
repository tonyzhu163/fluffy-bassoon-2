# Independent readback of the source-demonstration handoff

**The primary IIWA34 handoff completed 6/8 Kinova3 drawer starts, against 2/8 for tuned π0.5.** It rescued four of six tuned failures and retained both tuned successes. The difference was +50 percentage points, with a descriptive 95% paired-layout bootstrap interval of +12.5 to +87.5 points. The secondary IIWA35 handoff also completed 6/8. It rescued five tuned failures and lost one tuned success. The raw policy completed 3/8. All 32 native episodes and all 16 handbacks passed independent readback. [Compact native audit](audit/native.json) · [Original episode transport](evidence/native01/episodes.tar.gz)

## Outcomes on the eight paired starts

A number is the one-based control at which the native simulator returned reward 1 and terminated. A dash is a completed 300-control timeout with reward 0. Every preceding control had zero native reward and no terminal flag.

| Kinova3 layout | Raw π0.5 | Tuned π0.5 | IIWA34 → tuned | IIWA35 → tuned |
|---|---:|---:|---:|---:|
| 40 | — | — | 220 | 236 |
| 41 | 171 | 175 | 175 | 187 |
| 42 | 181 | 169 | 179 | — |
| 43 | — | — | 169 | 197 |
| 44 | — | — | — | — |
| 45 | — | — | 218 | 194 |
| 46 | 200 | — | 176 | 196 |
| 47 | — | — | — | 196 |
| **Successes** | **3/8** | **2/8** | **6/8** | **6/8** |

Each rescue denominator counts baseline failures. Each loss denominator counts baseline successes. Intervals use the 2.5th and 97.5th percentiles of all 8⁸ ordered resamples of whole paired layouts, calculated by multinomial counting. They describe sensitivity to the mixture of these eight fixed starts.

| Comparison | Rescued failures | Lost successes | Ties | Net difference | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| IIWA34 vs tuned, primary | 4/6 | 0/2 | 4 | +50 pp | +12.5 to +87.5 pp |
| IIWA35 vs tuned, secondary | 5/6 | 1/2 | 2 | +50 pp | 0 to +87.5 pp |
| IIWA34 vs raw | 3/5 | 0/3 | 5 | +37.5 pp | +12.5 to +75 pp |
| IIWA35 vs raw | 4/5 | 1/3 | 3 | +37.5 pp | −12.5 to +87.5 pp |
| Tuned vs raw | 0/5 | 1/3 | 7 | −12.5 pp | −37.5 to 0 pp |

Both hybrids rescued raw and tuned failures on 40, 43 and 45. Both restored row 46, which raw solved and tuned lost. IIWA34 retained row 42 and failed on 47. IIWA35 reversed those outcomes. All four modes failed on 44. The primary result supports using this recorded source prefix to change a frozen motor’s initial approach on these layouts. Two fixed donors on one reused drawer task leave generalization to other demonstrations, tasks and layouts open. [Per-layout actions, rewards and handbacks](audit/native.json)

## Checks on the distributed run

Baige Distributed Training job `job-mbgx2xjo8rqm` ran the qualified revision on one RTX 4090 and exited zero after 747 launcher seconds. I read all 32 distinct original episode files from shared storage on Linux cores 46/47 with GPUs hidden. The reader checked 7,739 native controls and 1,555 frozen-policy calls. All 17 success episodes reached their first native success at the final control. The remaining 15 reached native truncation at control 300. The reader independently reproduced the study writer’s outcome counts, rescue and loss rows and intervals. The study owner’s portable archive contains byte-identical copies of all 32 episodes. [Independent readback](handoff_readback.py) · [Launcher receipt](evidence/native01/exit.txt) · [Transport hashes](audit/transport.json)

- All eight release-row fingerprints were distinct. The four modes on every layout agreed on seed, BDDL and perturbation, physical and settled reset, target body, initial public and converted images/state, full noise bank and opening proposal. Every branch used its first reset attempt.
- All 840 IIWA34 prefix controls and 920 IIWA35 prefix controls matched the seven-channel source bytes at the underlying simulator step. All 16 hybrids reached handback. I regenerated all 60 noise slices per layout and checked each executed model query against the slice at the global five-control index.
- At every handback, both camera inputs and the state had changed since the preceding query. Their hashes matched the ordinary tuned conversion of the current target response. At the same global index, raw and tuned had the same noise as the hybrid, with different model inputs and different proposals on all 16 comparisons per baseline.
- I independently transformed every post-cut proposal and compared all seven outgoing float32 channels, including the final partial blocks. All 897 IIWA34 and 886 IIWA35 post-cut controls matched exactly. Every one differed from its uncorrected command. All 2,144 tuned-baseline controls matched too. Two separately rounded float32 products followed by addition reproduce the GPU host’s planar rotation throughout. The CPU qualifier matches its local BLAS arithmetic. No numerical tolerance was needed.

## Registered comparison

The primary comparison is the shortest successful IIWA recording, donor 34, against a standard tuned frozen π0.5 controller. The development-selected donor 35 is secondary. Each donor also has a comparison to raw π0.5 on the same eight Kinova3 starts. The estimand is the difference in native success by control 300 on the fixed, previously used release layouts 40–47. Each layout has one seed and contributes one paired difference per comparison. [Registered plan](PLAN.md)

Donor 34 finished in 171 source controls, the shortest of four successful IIWA episodes. Donor 35 was selected on four earlier Kinova3 layouts, including its rescue on 37. In a later five-layout CPU replay, donor 35 completed one start that raw also solved. Donor 34 completed one different start, layout 14, which raw failed. These are whole-tape results on earlier layouts. [Development matrix](../frozen-transfer-donor-20261124-r1/RESULT.md) · [Five-layout validation](../frozen-transfer-donor-validation-20261124-r1/RESULT.md)

## Native actions and observation path

Both the original cue archive and the derivative raw archive matched their pinned SHA-256 values. The two copies of each donor episode were byte-identical. I extracted all seven float32 channels from the actions the original IIWA worker actually sent. The installed PandaGripper maps negative to opening and positive to closing. Its signed, incremental opposing-finger command establishes the command semantics. The cut identifies a change in the recorded command and supplies no measurement of contact or completion of a physical phase. [Original cue archive](../frozen-transfer-cue-20261124-r1/evidence/native01/native-tapes.tar.gz) · [Raw archive manifest](../frozen-transfer-donor-20261124-r1/inputs/manifest.json)

| Donor | Initial grip | Last negative before switch | First positive grip | Zero-based switch index | Donor control count | First target control, one-based | Global query index, zero-based |
|---|---:|---:|---:|---:|---:|---:|---:|
| IIWA34 | −0.999990 | −1 | +0.992273 | 106 | 105 | 106 | 21 |
| IIWA35 | −1 | −1 | +0.993763 | 119 | 115 | 116 | 23 |

The release manifest identifies LIBERO-Pro2.0 Goal 3 BASE, with 300 relative 7-channel OSC controls at 20 Hz. The verified BDDL success predicate is the Akita black bowl inside the top cabinet region. The native arm wrapper passes the action to the simulator, reads its reward, and checks its success predicate. The audit hook copies the seven float32 values inside that underlying step and returns the native reward and terminal flags with a consecutive step index. The simulator uses mounted Kinova3 and a Panda gripper. [Step instrumentation](audited_worker.py) · [Physical arm environment](../libero-crossarm-20261120-r1/arm_env.py)

The prefix selector uses only the registered mode, the fixed donor tape and elapsed controls. The ordinary correction comes from a separate Kinova3 layout-11 passive tape with three valid five-control blocks and PR7 reader seed 41. I rehashed that tape and its prediction record. The correction is yaw 4.283361°, scale 0.920603 and gripper sign +1. Task predicates, simulator tool state, contact and object poses do not enter action selection. The Python wrapper retains complete simulator responses, including reset diagnostics and native success flags used by the evaluator. Source inspection establishes which fields the command selector and motor actually consume. Process isolation against reading these diagnostics was outside the design. [Policy interface](run.py) · [Reference support join](../frozen-transfer-progress-20261124-r1/contract.py)

Each branch creates a fresh interface and physically resets. The motor queries on both images and proprioception from the latest returned target observation. After the opening, tuned modes retarget end-effector position about that episode’s initial anchor. The wrapper independently converts the current public observation, compares its hashes to the actual input, then passes that input, the original instruction and the global noise slice to the shared frozen π0.5. The simulator hook hashes the actual post-step public observation. This links each query to the preceding physical response. The stored hashes and pinned code allow an independent check of that link. Recreating the model inputs from pixels would require raw images beyond the retained episode records. [Live recorder](run.py) · [Pinned motor implementation](../libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py)

## Separate whole-tape CPU condition

I verified the earlier 16 CPU case files, eight start files, full command hashes and executed prefixes for these two donors on layouts 40–47. IIWA34 completed 3/8 (41, 43, 46), and IIWA35 completed 2/8 (46, 47). Other tapes ran out with neither native terminal flag set, at control 171 or 184 respectively. By control 171, the counts were 3/8 and 0/8. The CPU and distributed runs shared the eight official layout fingerprints, seeds and pre-settle scenes. Their settled scene and physics hashes differed on all eight starts. The CPU tapes also ended on exhaustion, while each hybrid continued to native success or the 300-control horizon. The 3/8 and 2/8 CPU scores therefore cannot isolate the contribution of closed-loop handback from host and stopping-rule differences. [Original CPU case records](../frozen-transfer-donor-validation-20261124-r1/evidence/cases/) · [CPU validation](../frozen-transfer-donor-validation-20261124-r1/RESULT.md)

## Before launch

The first verifier recast a partial final block with a shorter matrix multiplication. A host BLAS can round that calculation differently from the five-row transform that produced the controls. Revision `ef165714d626305fa756e10fac2b5f7c45446f1c` converts all five proposals before slicing and saves each episode before verifying it. The two-file diff against the previous pin shows the same command-generation path. The independent reader also computes all five rows first and requires exact float32 bytes across every executed channel.

The corrected-pin qualifier `cpu02` exited zero. On layout 40, four matched openings led to 240 native simulator steps and 48 queries. The raw and tuned fixtures each stopped at five controls. Both hybrids executed the exact prefix and five further controls, reaching 110 and 120. Only the two handback queries used the genuine frozen π0.5 checkpoint. On each, the actual outgoing controls matched the tuned transform, differed from the uncorrected proposal, and used the new physical observation and the expected global noise slice. The independent reader rejected five edits made separately to copies of each donor record. The edits changed one bit in a donor control, reused a stale model input, restarted the noise clock, bypassed the tuned conversion or fabricated a reward. [Qualification audit](audit/qualification.json)

## Repeating the independent check

Run the reader on an available Baige Linux CPU pair with GPU visibility disabled. The reader uses NumPy plus CPU PyTorch to read the release bank. The study owner retains the full episode transport. The audit JSON records SHA-256 for each original episode and receipt, the code revisions, every matched start, handback and paired result. The following command reads the originals in shared storage.

```bash
root=/mowenjie/fluffy-bassoon-2
study=frozen-transfer-demonstration-20261124-r1
pin=ef165714d626305fa756e10fac2b5f7c45446f1c
src="$root/setups/symmeta-perceiver-followups-20261123-r1/src-${pin:0:12}"
env CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 46,47 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B \
    handoff_readback.py "$root/outputs/$study/gpu01" --cores 46,47 \
    --source "$src" --pin "$pin" --job job-mbgx2xjo8rqm
```
