"""Exercise the real Slurm-step CLI, resume path and checkpoint-on-signal boundary."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch


def command(output, updates, resume=None, core="gru"):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = [sys.executable, "-m", "fluffy_bassoon.meta_rl.train", "--source-commit", source,
              "--output-dir", str(output), "--updates", str(updates), "--num-envs", "2",
              "--width", "8", "--eval-trials", "4", "--eval-every", "2",
              "--checkpoint-every", "1", "--ppo-epochs", "1", "--minibatch-trials", "2",
              "--max-elapsed-seconds", "120", "--device", "cpu", "--core", core]
    if resume is not None:
        result += ["--resume", str(resume)]
    return result


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="CLI integration requires an allocated Linux step")
@pytest.mark.parametrize("bounded,motor",[(False,False),(True,False),(True,True)])
def test_cli_resume_matches_uninterrupted_training(tmp_path,bounded,motor):
    first, resumed, direct = (tmp_path / name for name in ("first", "resumed", "direct"))
    initializer=[]
    if motor:
        from fluffy_bassoon.evaluation_stages import sha256
        parent=tmp_path/'motor'
        args=command(parent,2)+['--task','reach','--horizon','2','--bounded-action-entropy','--mode','episode']
        result=subprocess.run(args,text=True,capture_output=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
        initializer=['--motor-warmstart',str(parent/'latest.pt'),'--motor-warmstart-sha256',sha256(parent/'latest.pt')]

    for args in (command(first, 2), command(resumed, 3, first / "latest.pt"), command(direct, 3)):
        if bounded:args += ["--task","reach","--horizon","2","--bounded-action-entropy"]
        args += initializer
        result = subprocess.run(args, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    a = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(direct / "latest.pt", map_location="cpu", weights_only=False)
    assert a["iteration"] == b["iteration"] == 3
    assert a["counters"]["environment_steps"] == b["counters"]["environment_steps"] == (48 if bounded else 24)
    for key in a["policy"]:
        torch.testing.assert_close(a["policy"][key], b["policy"][key], rtol=0, atol=0)
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["completed"] and summary["checkpoint"]["sha256"]
    config = json.loads((resumed / "config.json").read_text())
    assert config["resumed_from"]["path"] == str(first / "latest.pt")
    assert config["contract"]["optimization"]["bounded_entropy"] is bounded
    torch.testing.assert_close(a["optimizer"],b["optimizer"],rtol=0,atol=0)
    if motor:
        assert config['contract']['initialization']['pretraining_counters']['environment_steps']==32
        assert config['contract']['initialization']['source_iteration']==2
        assert config['motor_warmstart_path']==str(parent/'latest.pt')


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="Signal qualification requires an allocated Linux step")
def test_usr1_finishes_with_a_complete_trial_boundary_checkpoint(tmp_path):
    output = tmp_path / "signalled"
    log = (tmp_path / "child.log").open("w")
    process = subprocess.Popen(command(output, 10000), stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while not (output / "latest.pt").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert (output / "latest.pt").exists(), (tmp_path / "child.log").read_text()
        process.send_signal(signal.SIGUSR1)
        assert process.wait(timeout=60) == 0, (tmp_path / "child.log").read_text()
        summary = json.loads((output / "summary.json").read_text())
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        assert not summary["completed"] and saved["boundary"] == "between_trials"
        assert saved["iteration"] == summary["iteration"]
        assert not (output / "latest.partial").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
        log.close()


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="CLI integration requires an allocated Linux step")
@pytest.mark.parametrize("kind", ["ctm", "looped_transformer", "transformer_xl"])
def test_cli_records_effective_control_and_state_bytes(tmp_path, kind):
    output = tmp_path / kind
    args = command(output, 1, core=kind) + ["--stm-length", "3", "--synapse-depth", "2"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((output / "config.json").read_text())
    assert config["contract"]["core"] == kind
    if kind == "ctm":
        assert config["contract"]["core_config"]["layers"] == 0
        assert config["contract"]["core_config"]["ltm_tokens"] == 0
        assert config["retained_state_bytes_per_lane"] == 8 * (3 + 3) * 4
    elif kind == "transformer_xl":
        assert config["contract"]["core_config"] == dict(width=8, ticks=1, memory_length=64, layers=2, heads=4)
        assert config["retained_state_bytes_per_lane"] == 2 * 64 * 8 * 4 + 8
    else:
        assert config["contract"]["core_config"]["memory_tokens"] == 8
        assert config["retained_state_bytes_per_lane"] == 8 * 8 * 4


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="CLI integration requires an allocated Linux step")
def test_bernoulli_cli_records_rewards_and_diagnostic_regret(tmp_path):
    output = tmp_path / "bernoulli"
    args = command(output, 1) + ["--task", "bernoulli", "--arms", "3", "--attempts", "8",
                                 "--reward-offset", ".5", "--value-coef", ".01"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert config["contract"]["arms"] == 3
    assert config["contract"]["optimization"]["reward_offset"] == .5
    assert config["contract"]["optimization"]["value_coef"] == .01
    assert config["contract"]["optimization"]["normalize_advantages"] is True
    normal = summary["last_evaluation"]["normal"]
    assert "success_by_attempt" not in normal and len(normal["reward_by_attempt"]) == 8
    assert normal["cumulative_expected_regret"] >= 0


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="CLI integration requires an allocated Linux step")
def test_transformer_xl_cli_exact_resume_and_read_only_evaluation(tmp_path):
    from fluffy_bassoon.evaluation_stages import sha256
    first, resumed, direct = (tmp_path / name for name in ('first', 'resumed', 'direct'))
    for args in (command(first,2,core='transformer_xl'),
                 command(resumed,3,first/'latest.pt',core='transformer_xl'),
                 command(direct,3,core='transformer_xl')):
        result=subprocess.run(args,text=True,capture_output=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
    a,b=(torch.load(p/'latest.pt',map_location='cpu',weights_only=False) for p in (resumed,direct))
    assert a['iteration']==b['iteration']==3
    for key in a['counters']:
        # Resume performs one extra read-only evaluation: four conditions of four 4-step trials.
        extra=64 if key in ('evaluation_steps','evaluation_policy_steps') else 0
        assert a['counters'][key]==b['counters'][key]+extra
    torch.testing.assert_close(a['policy'],b['policy'],rtol=0,atol=0)
    torch.testing.assert_close(a['optimizer'],b['optimizer'],rtol=0,atol=0)
    path=direct/'latest.pt';digest=sha256(path)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result=subprocess.run([sys.executable,'-m','fluffy_bassoon.meta_rl.evaluate',
        '--source-commit',source,'--checkpoint',str(path),'--output-dir',str(tmp_path/'eval'),
        '--policy-seeds','2000000000'],text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/'eval/summary.json').read_text())
    assert report['model_tensors_unchanged'] and report['completed_run_greedy_reference_matched']
    assert sha256(path)==digest
