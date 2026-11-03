"""One allocated-Linux offline exporter/privacy/resume check, no model work."""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

from fluffy_bassoon.meta_rl.joint_steering_metrics import MetricPublisher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if sys.platform != "linux":
    raise RuntimeError("Run qualification on allocated Linux only")
root = args.output
root.mkdir(parents=True, exist_ok=False)
sentinel = "NONSECRET_PRIVACY_QUALIFICATION_SENTINEL"
profile = dict(arm="velocity", updates=50, episodes_per_update=384, lanes_per_worker=48,
    native_env=dict(never_export=sentinel), wandb=dict(enabled=False,
        group="steering-metrics-offline-qualification", run_id="metricsqualify50"))
status = dict(phase="train", iteration=1, sequence=1, elapsed_seconds=3.,
    escrow=dict(charged=dict(train=100, eval=0), pending={}),
    summary=dict(episodes=2, macro_ID_success_once=.5,
        per_task=dict(OpenDrawer=dict(episodes=2, success_once=1, success_at_end=0))),
    ppo=dict(loss=.2, actor_loss=.1, critic_loss=.2, optimizer_steps=2,
        final_metrics=dict(joint_gaussian_kl=.01, clip_fraction=.1, explained_variance=.3)))
first = MetricPublisher(root, profile)
first.publish(status)
first.close()
identity = json.loads((root / "METRICS-IDENTITY.json").read_text())
profile["wandb"]["enabled"] = True
second = MetricPublisher(root, profile)
second.publish(dict(status, phase="dev", iteration=5, sequence=2))
deadline = time.monotonic() + 45
exported = root / "METRICS-EXPORTED.json"
while time.monotonic() < deadline:
    if exported.exists() and json.loads(exported.read_text())["sequence"] == 2:
        break
    failed = root / "METRICS-EXPORT-STATUS.json"
    if failed.exists() and json.loads(failed.read_text()).get("initialized") is False:
        break
    time.sleep(.5)
second.close()
state = json.loads((root / "METRICS-EXPORT-STATUS.json").read_text())
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
config = (root / "METRICS-EXPORT.json").read_text()
assert sentinel not in config and "native_env" not in config
assert json.loads((root / "METRICS-IDENTITY.json").read_text()) == identity
assert [row["sequence"] for row in rows] == [1, 2]
assert rows[0]["metrics"]["train/ppo/loss"] == .2
assert rows[0]["metrics"]["train/current/joint_gaussian_kl"] == .01
assert state["initialized"] and state["mode"] == "offline"
assert json.loads(exported.read_text())["sequence"] == 2
result = dict(completed=True, hostname=socket.gethostname(), mode="offline", SDK=state["SDK_version"],
    stable_id=identity["run_id"], local_events=2, SDK_submitted_events=2,
    arbitrary_profile_fields_exported=False, model_computation=False, physical_controls=0,
    authenticated_online_project_qualified=False)
(root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
