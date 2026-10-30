"""Resource/scientific contracts of the two independent packed qualification forks."""
import importlib.util
from pathlib import Path

_path=Path(__file__).parents[1]/'scripts/diagnostics/robotwin_packed_ppo_probe.py'
_spec=importlib.util.spec_from_file_location('packed_probe',_path)
probe=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(probe)


def test_packed_forks_have_disjoint_ledgers_and_unchanged_learning_batches(tmp_path):
    values=probe.commands(Path('/pinned/worktree'),tmp_path,'a'*40,python='/python')
    assert set(values)=={'ppo','ppo_reset'}
    def arg(command,name):return command[command.index(name)+1]
    for method,command in values.items():
        assert arg(command,'--methods')==method
        assert arg(command,'--collector-lanes')=='6'
        assert arg(command,'--ppo-trials-per-update')=='6'
        assert arg(command,'--train-episodes')=='4'
        assert arg(command,'--source-iterations')=='1'
        assert arg(command,'--source-steps')==arg(command,'--campaign-max-actions')=='10800'
        assert arg(command,'--campaign-root')==str(tmp_path/('ledger-'+method))
        assert arg(command,'--native-episodes')==arg(command,'--source-validation-episodes')=='0'
        assert '--stage-source-validation-only' in command and '--resume' not in command
        assert arg(command,'--learning-rate')=='1e-6' and arg(command,'--ppo-epochs')=='5'
    assert probe.ACTIONS_PER_ARM*len(values)==21600


def test_gpu_telemetry_respects_attested_uuid(monkeypatch):
    calls=[]
    def fake(command,**kwargs):
        calls.append(command);return '76800, 98304, 87\n'
    monkeypatch.setattr(probe.subprocess,'check_output',fake)
    row=probe.sample_gpu('GPU-owned')
    assert '--id=GPU-owned' in calls[0]
    assert row['used_gib']==75 and row['total_bytes']==96*(1<<30)
    assert row['utilization_percent']==87
