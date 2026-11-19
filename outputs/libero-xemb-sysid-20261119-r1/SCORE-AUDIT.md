# Independent native score readback

November 19, 2026. Static verification only, 0 GPU-hours. No native outcome was scored during preparation. The scorer reads final Baige JSON and writes its report to stdout. It uses the Python standard library and needs no simulator, checkpoint or GPU. [Scorer](score_poc.py)

## Run on the archived evidence

Set `score` to the deployed scorer path and each attempt variable below to its actual Baige attempt directory. Gate 1 and the pilot can be read separately or in a single invocation. Keep each producer's revision from its own `launch.txt`. The report fingerprints each supplied JSON file independently.

```bash
python3 "$score" --gate1 "$gate_attempt/gate1/gate1-summary.json"

python3 "$score" --pilot "$eval_attempt/live/live-eval-summary.json" \
  --archive "$collect_attempt/archive/archive.json" \
  --fit "$train_attempt/train/offline-audit.json"
```

The optional archive and fit inputs check the support schedule, distinct support/query physics, per-arm acquisition counts, archive linkage, fitting seed and final update count. Omitting one leaves its verification field null. Exit 0 means the readback completed, including a valid negative result. Exit 2 identifies invalid evidence, a mismatch with published counts or a disagreement with the archived decision. Smoke and progress files cannot produce a decision.

## What to read

- **Gate 1.** The registered denominator is 48 matched pairs on eight tasks. Passing requires at least 36 oracle successes, at least 24 more successes than nominal, and no aggregate regression on the eight nominal-embodiment pairs. Every episode retains its final simulator success, return and executed controls. Success is read from the native flag. A reward trace must contain one finite value per executed control and sum to the archived return. The `95388e6e` layout supplies only `archived_sum_only` returns. Its per-control rewards and full opening proposals are unavailable. [Gate contract](PLAN.md) · [Motor evidence fields](MOTOR-AUDIT.md)
- **Parity.** Within each task, every arm and embodiment must agree on the full query reset, including physics, camera/state digests, released prompt, horizon, noise seed and noise-bank digest. Later records also compare the opening frozen proposal across arms that query the motor at control zero. The online arm spends its prefix inside the query horizon. These checks use archived digests. They establish pairing at reset and leave subsequent renderer drift open. [Motor](motor_iface.py) · [Live writer](eval_live.py)
- **First pilot.** Read `first_poc_outcome_rule` and `first_poc_witnesses`. A witness is one shifted case on which a retained active or passive reader and the oracle succeed, while nominal, cleared and swapped all fail. BASE, the held-out three-factor composition and the unseen 45° yaw each have their own arm counts, returns, paired success gains in percentage points, return gains, wins, losses and ties under `groups`. The case records show actual support exposure and online-prefix costs. Missing arms are named. The cleared and swapped controls use the active checkpoint, which limits attribution of a passive-reader rescue. [Pilot rule](FIRST-POC.md)
- **Interpretation.** Compare each witness with analytic proprioception, train-calibrated analytic vision, joint, non-equivariant and online controls on that same case. A visual tie leaves the need for learning open. Oracle failure leaves realizability unresolved on that case. A reader failure with oracle success leaves a realizable gap. The yaw intervention constructs a coordinate mismatch with fixed cameras, scale changes all three translation commands, and free-space jaw motion exposes gripper polarity. A one-task rescue supports replication and gives no evidence of contact adaptation. [Pilot contract](FIRST-POC.md)

Mac CPython 3.9.6 compiled the scorer in memory. Static source inspection covered the Gate 1 writer at `95388e6e`, the pilot evaluator and support/fit writers at `65c83998`, and the registered development task identities. The scorer was not executed on Mac.
