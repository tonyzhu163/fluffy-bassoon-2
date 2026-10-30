# Fresh-process native K8 wrapper

`python -m fluffy_bassoon.meta_rl.robotwin_history_live` is the live wrapper
for the immutable fixed100 native panel. The execution owner supplies the
reviewed allocation, exact source commit, external source paths and a fresh
evaluation ledger. It launches no allocation and performs no SGD.

The parent runs the focused wrapper/adapter fake tests before actions, admits
all four actual portable endpoints and source SAC5 on CPU, then invokes the
existing Baige evaluation GPU probe. It preserves scheduler visibility, records
the resulting GPU map and runs18 sequential fresh Python sessions: native
Gaussian for both tasks first, then the remaining eight conditions for both
tasks. Every child executes ONE K8 unit; no child runs another task or unit.

Each child uses qualified DB `RobotwinNoiseEnv` construction, source asset and
external Git checks, final-camera capture, the saved capture qualification,
CUDA/render binding and native-noise parity. Parity uses one zero-action reset,
then the declared eight data episodes: at most9 nominal accepted resets per
child,144 data episodes total. Internal reset attempts and actual scene
creations remain recorded separately. This18-process operational refinement
supersedes the old plan's nine-process description without changing its
immutable cases, conditions, model hashes or57,600-action cap.

Only the existing evaluation adapter meters physical actions. The environment
is raw, with no `CampaignEnv` double wrapper. Each child opens the same separate
57600-action target ledger exclusively, writes raw corpus and episode files,
checks closed K8 accounting, closes the environment, then emits its immutable
unit receipt. Parent checks the child exit code, receipt/result hashes, exact
paired case order and cumulative journal after every unit. A native fault,
unresolved intent, mismatch, timeout, nonzero exit or close failure halts the
panel. There is no automatic retry or lost-physics replay.

Children use new sessions. Parent supervises each bounded subprocess and cleans
only its captured PGID/SID, with bounded TERM→KILL cleanup; unexpected session
identity is refused. Parent SIGTERM/interrupt unwinds the same cleanup path.
Source jobs and other users' processes are never targeted.

## Owner command

Run inside the admitted one-GPU allocation, from the exact clean execution
checkout. Use the existing qualified Python environment and the actual
external RoboTwin/RLinf directories. Keep runtime output and ledger OUTSIDE
the source checkout. The outer allocation/launcher remains bounded to90minutes
including startup, failed attempts and final teardown.

```bash
export PYTHONPATH="$TASK_CODE/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
timeout --signal=TERM --kill-after=10s 5390 "$TASK_PYTHON" \
  -m fluffy_bassoon.meta_rl.robotwin_history_live \
  --plan "$TASK_CODE/outputs/robotwin-native-evaluation-plan-20261030-r1/plan.json" \
  --plan-sha b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c \
  --output "$TASK_OUT/evaluation" --campaign-root "$TASK_OUT/evaluation-ledger" \
  --campaign-id "$TASK_RUN" --source-commit "$TASK_PIN" \
  --expected-hostname "$(hostname)" \
  --robotwin-root "$TASK_ROBOTWIN" --rlinf-root "$TASK_RLINF" \
  --wall-seconds 5300 --prior-gpu-seconds 0
```

`TASK_OUT` may contain the owner's auxiliary logs; its `evaluation` subfolder
must be fresh. The wrapper keeps at least60 seconds for finalization and
requires at least300 more seconds before starting another K8 child. A child
timeout can leave uncertain physical cost and a partial trial; it halts the
panel rather than presenting that trial as complete.

## Results and explicit safe recovery

`evaluation/{execution,admission,gpu-probe,progress,summary}.json`, test/probe
logs, per-unit stdout/process receipts and `units/*/{qualification,unit-receipt}.json`
bind execution, models, runtime and actual costs. Raw corpus, renderer evidence,
atomic episode files and the complete K8 `trial/result.json` stay under each
unit. Parity's cached decode plus direct native prediction are counted as two
inference calls and zero policy actions. A native renderer/reset failure never
becomes a fabricated free transition.

A planned wall stop between completed units returns `status=wall_boundary`.
An explicitly reviewed new allocation may supply `--resume-manifest` and its
`--resume-sha`, a NEW output subfolder, the SAME code/plan/ledger/campaign and
`--prior-gpu-seconds` covering prior actual charged GPU time. Set the new
`--wall-seconds` no greater than5400 minus that prior charge; the external
allocation must honor the same residual budget. Completed units and their
native case receipts are rehashed and reused without executing again.

Failed/unknown-outcome manifests are refused by automatic recovery. Preserve
their partial artifacts and charged/uncertain journal and diagnose separately.
Launcher elapsed time is a lower bound on allocated GPU cost; the execution
owner also records actual scheduler duration and overhead. A complete panel
requires `status=complete`,18 verified K8 receipts and a closed physical ledger.
One K8 trial per condition/target remains a small development screen.
