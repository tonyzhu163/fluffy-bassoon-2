"""Explicit scalar telemetry; local boundaries never depend on remote logging.

Only the master calls this module. The optional W&B exporter is a separate
CPU-only process and receives a whitelisted configuration, never the runtime
profile, native environment, observations, checkpoints or source files.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def scalar_metrics(status):
    """An explicit scalar allowlist; no recursive export of an arbitrary profile."""
    phase = status["phase"]
    result = {"ppo/update": status["iteration"], "event/sequence": status["sequence"]}
    def add(name, value):
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = value
    for name in ("elapsed_seconds", "collection_and_learning_seconds", "physical_controls_this_event", "controls_per_second",
                 "projected_all_attempt_gpu_hours", "remaining_gpu_hours", "learning_rate",
                 "steering_std_mean", "steering_std_min", "steering_std_max"):
        add("runtime/" + name, status.get(name))
    escrow = status.get("escrow", {})
    for category, value in escrow.get("charged", {}).items():
        if category in ("train", "eval"):
            add("counters/" + category + "_physical_controls", value)
    add("counters/physical_controls", sum(escrow.get("charged", {}).values()))
    add("counters/unresolved_calls", len(escrow.get("pending", {})))
    summary = status.get("summary", {})
    for name in ("episodes", "macro_ID_success_once", "macro_ID_success_at_end",
                 "macro_OOD_success_once", "macro_OOD_success_at_end"):
        add(phase + "/" + name, summary.get(name))
    for task, row in summary.get("per_task", {}).items():
        episodes = row["episodes"]
        add(phase + "/task/" + task + "/episodes", episodes)
        for key in ("success_once", "success_at_end"):
            if episodes:
                add(phase + "/task/" + task + "/" + key, row[key] / episodes)
    ppo = status.get("ppo", {})
    for name in ("loss", "actor_loss", "critic_loss", "entropy", "approximate_kl", "clip_fraction",
                 "samples", "samples_processed", "optimizer_steps", "epochs_completed", "epochs_started",
                 "return_mean", "return_std", "advantage_mean", "advantage_std", "elapsed_seconds",
                 "coefficient_rms", "coefficient_saturation_fraction", "positive_reward_count", "negative_reward_count"):
        add("train/ppo/" + name, ppo.get(name))
    for name in ("loss", "actor_loss", "critic_loss", "entropy", "approximate_kl", "joint_gaussian_kl",
                 "clip_fraction", "explained_variance"):
        add("train/current/" + name, ppo.get("final_metrics", {}).get(name))
    epochs = ppo.get("epoch_metrics", [])
    if epochs:
        add("train/gradient_norm_max", max(row["gradient_norm_max"] for row in epochs))
        add("train/gradient_norm_mean", sum(row["gradient_norm_mean"] * row["optimizer_steps"]
            for row in epochs) / max(1, ppo["optimizer_steps"]))
    if ppo.get("samples") is not None and summary.get("episodes"):
        add("train/valid_chunk_fraction", ppo["samples"] / (12 * summary["episodes"]))
    paired = status.get("paired_development", {}).get("ID", {})
    for metric in ("success_once", "success_at_end"):
        row = paired.get(metric, {})
        for name in ("gain", "exact_two_sided_mcnemar_p", "before", "after", "after_only", "before_only"):
            add("dev/paired/" + metric + "/" + name, row.get(name))
        for name, value in zip(("ci_low", "ci_high"), row.get("paired_task_stratified_bootstrap_95ci", [])):
            add("dev/paired/" + metric + "/" + name, value)
    resources = status.get("resources", {})
    for name in ("container_RAM_GiB", "container_peak_RAM_GiB"):
        add("resources/" + name, resources.get(name))
    for index, row in enumerate(resources.get("GPUs", [])):
        for key in ("memory_used_MiB", "utilization_percent"):
            add(f"resources/GPU{index}/" + key, row.get(key))
    return result


def resource_snapshot():
    result = {}
    for filename, name in (("memory.current", "container_RAM_GiB"),
                           ("memory.peak", "container_peak_RAM_GiB")):
        try:
            result[name] = int((Path("/sys/fs/cgroup") / filename).read_text()) / 2**30
        except (OSError, ValueError):
            pass
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True)
        result["GPUs"] = [dict(zip(("memory_used_MiB", "utilization_percent"),
                                    map(float, line.split(",")))) for line in query.stdout.strip().splitlines()]
    except Exception as error:
        result["GPU_metric_unavailable"] = type(error).__name__
    return result


class MetricPublisher:
    def __init__(self, run, profile):
        self.root, self.process, self.console = Path(run), None, None
        options = profile.get("wandb", {})
        group = options.get("group", "joint-steering-scale50-20261103-r1")
        identity = dict(arm=profile["arm"], group=group,
            run_id=options.get("run_id", hashlib.sha256((group + "/" + profile["arm"]).encode()).hexdigest()[:16]))
        self.identity = identity
        try:
            path = self.root / "METRICS-IDENTITY.json"
            if path.exists() and json.loads(path.read_text()) != identity:
                raise ValueError("Metric identity differs on resume")
            _json(path, identity)
            if not options.get("enabled", False):
                return
            configuration = dict(**identity, project=options.get("project", "joint-steering-offline"),
                entity=options.get("entity"), online_authorized=bool(options.get("online_authorized", False)),
                config=dict(arm=profile["arm"], updates=profile["updates"],
                    episodes_per_update=profile["episodes_per_update"], lanes_per_GPU=profile["lanes_per_worker"],
                    horizon=600, frozen_VLM=True, frozen_action_expert=True))
            _json(self.root / "METRICS-EXPORT.json", configuration)
            marker = self.root / "METRICS-EXPORT-STOP"
            if marker.exists():
                marker.unlink()
            environment = {key: value for key, value in os.environ.items()
                if key in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PYTHONPATH", "TMPDIR",
                           "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE",
                           "STEERING_WANDB_CREDENTIAL_FILE") or key.startswith("WANDB_")}
            environment.update(CUDA_VISIBLE_DEVICES="", WANDB_CONSOLE="off", WANDB_DISABLE_GIT="true",
                WANDB_DISABLE_CODE="true", WANDB_MODE="online" if configuration["online_authorized"] else "offline")
            self.console = (self.root / "metrics-exporter-console.log").open("a")
            self.process = subprocess.Popen([sys.executable, "-B", "-m", __name__, "--directory", str(self.root)],
                env=environment, stdout=self.console, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as error:
            self._error(error)

    def _error(self, error):
        try:
            _json(self.root / "OPTIONAL-METRICS-ERROR.json", dict(error_type=type(error).__name__,
                nonfatal=True, local_checkpoints_and_event_metrics_authoritative=True))
        except OSError:
            pass

    def publish(self, status):
        try:
            row = dict(**self.identity, phase=status["phase"], sequence=status["sequence"],
                       unix=time.time(), metrics=scalar_metrics(status))
            with (self.root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as error:
            self._error(error)

    def close(self):
        try:
            (self.root / "METRICS-EXPORT-STOP").touch()
            if self.process is not None:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=2)
        except Exception as error:
            self._error(error)
        finally:
            if self.console is not None:
                self.console.close()


def export(directory):
    root = Path(directory)
    try:
        configuration = json.loads((root / "METRICS-EXPORT.json").read_text())
        credential_path = os.environ.get("STEERING_WANDB_CREDENTIAL_FILE")
        if configuration["online_authorized"] and credential_path:
            credential = Path(credential_path)
            if credential.stat().st_mode & 0o077:
                raise PermissionError("W&B machine-local credential permissions are too broad")
            os.environ["WANDB_API_KEY"] = credential.read_text().strip()
        import wandb
        fields = wandb.Settings.model_fields
        privacy = dict(console="off", disable_git=True, disable_code=True,
            x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True, x_disable_viewer=True)
        if not set(privacy) <= set(fields):
            raise ValueError("Unsupported W&B privacy settings")
        authenticated = "WANDB_API_KEY" in os.environ or (Path.home() / ".netrc").is_file()
        mode = "online" if configuration["online_authorized"] and authenticated else "offline"
        settings = wandb.Settings(**privacy, init_timeout=15)
        tracking = wandb.init(id=configuration["run_id"], project=configuration["project"],
            entity=configuration["entity"], group=configuration["group"], name=configuration["arm"],
            resume="allow", mode=mode, config=configuration["config"], dir=str(root), settings=settings)
        tracking.define_metric("ppo/update")
        tracking.define_metric("counters/physical_controls")
        tracking.define_metric("*", step_metric="ppo/update")
        _json(root / "METRICS-EXPORT-STATUS.json", dict(mode=mode, SDK_version=wandb.__version__,
            run_id=configuration["run_id"], initialized=True, nonfatal=True))
        last = 0
        state = root / "METRICS-EXPORTED.json"
        if state.exists():
            last = int(json.loads(state.read_text())["sequence"])
        while True:
            path = root / "metrics.jsonl"
            if path.exists():
                # Ignore an in-progress final append; complete records end in newline.
                for line in path.read_text().split("\n")[:-1]:
                    row = json.loads(line)
                    if row["sequence"] > last:
                        tracking.log(row["metrics"], step=row["sequence"])
                        last = row["sequence"]
                        _json(state, dict(sequence=last, run_id=configuration["run_id"],
                            submitted_to_SDK=True, remote_delivery_confirmed=False))
            if (root / "METRICS-EXPORT-STOP").exists():
                break
            time.sleep(2)
        tracking.finish()
    except Exception as error:
        _json(root / "METRICS-EXPORT-STATUS.json", dict(initialized=False,
            error_type=type(error).__name__, nonfatal=True, local_metrics_retained=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    export(parser.parse_args().directory)
