# PPO scalar metric relay

User request: periodically copy Baige metrics to this Mac, then upload them to
the existing Weights & Biases project. Training remains unchanged.

`scripts/diagnostics/sync_ppo_metrics.py` takes a read-only snapshot over SSH.
It mirrors complete JSONL scalar events, stable run identities, a small allowed
training configuration and the progress counters. Snapshots are hash checked.
Compatible successor attempt folders under the dated scale50 campaign are
discovered on later runs. Remote checkpoints, videos, observations, source,
console logs and SDK metadata are excluded.

The optional local uploader resumes the existing run IDs and uses the recorded
event sequence as the W&B history step. It confirms event/update counters through
W&B history readback before advancing the local upload cursor. Repeated invocations
are serialized by a file lock. Mirroring succeeds independently of cloud access.
Both current source runs have five metric events through PPO update 2, totaling
32,857 bytes; their hashes and counters are in `MIRROR-STATUS.json`.

Routing and authentication are machine-local in
`~/.config/fluffy-bassoon-2/metrics-relay.json`, outside this repository.
The local SDK environment uses uv and the same W&B version as the source exporter.
Run from the repository root:

```sh
/Users/mowenjie/.cache/fluffy-bassoon-2-metrics/venv/bin/python \
  scripts/diagnostics/sync_ppo_metrics.py \
  --directory outputs/ppo-metrics-relay-20261103-r1
```

Add `--local-only` to mirror without contacting W&B. `STATUS.json` records
the latest outcome, `MIRROR-STATUS.json` the source snapshots, and
`UPLOAD-STATE.json` only remotely confirmed upload cursors.

The active Codex heartbeat `sync-ppo-metrics-through-this-mac` invokes the relay
every 20 minutes while the Mac is awake and the app is running. Ordinary
successful syncs and unchanged failures stay quiet. It reports the first verified
cloud sync, a new actionable failure or completion.

## Initial verification

- SSH snapshots of both real metric streams copied successfully; all complete
  lines passed identity, finite-scalar, digest and sequence checks.
- Static Python parsing passed. No model, simulator, GPU job or training change.
- W&B's current official SDK identifies `https://api.forge.coreweave.com` as its
  API endpoint. Both that endpoint and the older `https://api.wandb.ai` returned
  HTTP 403 to authenticated and unauthenticated checks from this Mac.
- Feilian was inspected in the UI. It exposes VPN routing and showed both
  partial-site (极速) and all-site (全局) modes. A connected state was observed,
  but it did not establish working W&B API access. Original 极速 mode was restored.
- An actual local SDK upload attempt failed with `CommError`; no cloud upload
  or dashboard freshness is claimed. Metric files remain locally available.

Primary endpoint source:
[W&B SDK URL configuration](https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/urls.py).
