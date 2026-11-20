# Frozen π0.5 transferred to the simulated UR5e. The learned correction lost two successes.

On two LIBERO-Pro2.0 development tasks, unmodified π0.5 succeeded on all four UR5e starts. Training a history encoder and an action adapter on Panda and Sawyer yielded two successes on those same starts. Changing the history to another arm's made both of those successes disappear. Clearing history recovered all four. The experiment demonstrates that the controller uses information from practice. It gives no evidence of a task-success gain from this form of adaptation. [Paired outcomes](evidence/gpu01/pilot/status.json)

| UR5e controller | Bowl onto stove, starts 13 and 14 | Bowl next to ramekin onto plate, starts 13 and 14 | Total |
| --- | --- | --- | --- |
| Frozen π0.5 | 2/2 | 2/2 | **4/4** |
| Adapter with UR5e history | 0/2 | 2/2 | **2/4** |
| Same adapter with empty history | 2/2 | 2/2 | **4/4** |
| Same adapter with Panda or Sawyer history | 0/2 | 0/2 | **0/4** |
| Independently calibrated UR5e action correction | 2/2 | 1/2 | **3/4** |

Each column uses two distinct released starts. All five controllers on a given start shared the physical reset, initial images, instruction, flow noise and first frozen proposal. Each then received its own live observations. An independent CPU readback checked the native rewards for all 20 episodes against the recorded outcomes and aggregate counts. The official test split stayed sealed. [Case records](evidence/gpu01/pilot/pairs.json) · [Readback](evidence/posthoc01/audit.json)

## What went into the frozen policy's action

The seven-joint Panda, seven-joint Sawyer and six-joint UR5e ran their separate MuJoCo arm models with the same two-finger gripper and world-frame Cartesian controller. In stage one, the encoder used public commands and measured changes in wrist pose and jaw opening to predict the response to fresh commands. It trained on Panda and Sawyer for 300 updates. In stage two, frozen π0.5 supplied proposals from eight source rollouts. The encoder and action adapter trained together for 400 updates to match commands derived from each source arm's measured motion. Both source teachers labelled every proposal bank. The targets represent local calibration, without expert demonstrations or task-reward optimization. The frozen motor consumed the images, robot state and instruction. The reader consumed the command–motion history. [Fit and source pairings](evidence/gpu01/pilot/fit.json) · [Proposal banks](evidence/gpu01/pilot/proposal-banks.json)

At evaluation the reader compressed 16 transitions from a separate practice episode into 16 numbers. The adapter used these numbers to predict a correction to the six Cartesian components of each proposed action. The gripper command passed through. All weights stayed fixed during testing. The run saved the checkpoint before opening UR5e calibration. The holdout applies to both new training stages. The original π0.5 checkpoint's exposure to UR5e is unknown here. [Training and evaluation code](pilot.py) · [Adapter](learner.py) · [Checkpoint](evidence/gpu01/pilot/reader-adapter.pt)

## Could we identify the arm from the latent?

A post hoc diagnostic extracted the frozen reader's embeddings from two labelled 16-transition calibration histories per arm on Goal 4 and Goal 5. We used their per-arm means as references. Across the eight separate development histories, the nearest mean identified **3/8** arms. The probe called three of the four UR5e histories Panda and the fourth Sawyer. It identified 2/2 Panda and 1/2 Sawyer histories. The probe had a labelled UR5e reference available only for this diagnostic. The policy received no such reference. This small panel gives no evidence that arm identity is reliably readable with this rule. Other ways to decode the latent remain untested. [Diagnostic code](evidence/posthoc01/audit.py) · [Distances and predictions](evidence/posthoc01/audit.json)

Correct and wrong histories used identical commands on the same task and starting layout. Swapping the measured response changed the opening arm commands on all four query starts and lost both plate-placement successes. The source-training losses also fell. The stage-two action MSE on a sampled training batch was 0.000105 with source history and 0.0115 with empty history. That fitting pattern did not predict a held-out task advantage. [Opening-action differences](evidence/posthoc01/audit.json) · [Training diagnostics](evidence/gpu01/pilot/fit.json)

## What the controls establish

On fresh command probes, each arm's local response model predicted both translation and rotation better than constant drift. On the independent native admission start, both frozen and calibrated Panda succeeded, and both Sawyer controllers failed. Calibrated UR5e solved the stove task and failed the plate task. The prespecified combined feasibility check therefore failed on Sawyer. On the four evaluation starts the independently calibrated controller also lost one success relative to the raw motor. Accurate free-space response predictions alone did not guarantee good task actions. [Motion checks](evidence/gpu01/pilot/responses.json) · [Admission controls](evidence/gpu01/pilot/gate.json)

The raw motor's 4/4 leaves no room to measure improvement on these four starts. The two losses still measure harm. The shared Cartesian controller and common gripper may already absorb enough arm-level variation for these tasks. One way to distinguish that explanation from an insufficient adapter is to qualify a separate development condition where the frozen policy repeatedly fails and an independently specified correction repeatedly succeeds, then freeze that condition and compare histories on fresh starts. Hardware verification remains deferred.

Baige job `job-475bmodtjstu` finished successfully on one Pro6K in 7 min 51 s (0.131 GPU-hours). Training and evaluation used revision `e2b43d659ace204c7dd3499b898ff6536204bc21`, fit seed 41. Checkpoint SHA-256 is `507305af964d8796621455009123106ed6953e95c08a535cd087bae72edca496`. The Linux tapes, reset records, GPU probe, logs, checkpoint and diagnostic are archived on shared storage in `/mowenjie/fluffy-bassoon-2/outputs/libero-crossarm-20261120-r1/crossarm-sim-gpu01-evidence.tar.gz`. [Configuration](evidence/gpu01/pilot/config.json) · [Scheduler confirmation](baige-job.jpg)
