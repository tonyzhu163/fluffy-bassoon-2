"""Model admission can be reused for an environment-only change, never a motor change."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('change', ['environment', 'motor', 'patch', 'incomplete', 'loader'])
def test_reuse_keeps_native_loader_dependencies_fixed(tmp_path, monkeypatch, change):
    path = Path(__file__).resolve().parents[1]/'scripts/diagnostics/maniskill_pi05_baige_gates.py'
    spec = importlib.util.spec_from_file_location('native_load_proof_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    old, new = tmp_path/'old', tmp_path/'new'
    names = ['src/fluffy_bassoon/rl/rlinf_checkpoint_compat.py',
             'scripts/diagnostics/qualify_maniskill_checkpoint_heads.py',
             'patches/rlinf-e1a82ebf-flow-noise-keys.patch']
    for root in (old, new):
        for name in names:
            (root/name).parent.mkdir(parents=True, exist_ok=True)
            (root/name).write_text('fixed dependency')
        (root/'src/fluffy_bassoon/rl/maniskill_pi05.py').write_text(
            'def model_config():\n    return 1\ndef env_config():\n    return 1\n')
    current = new/'src/fluffy_bassoon/rl/maniskill_pi05.py'
    current.write_text(current.read_text().replace('def env_config():\n    return 1', 'def env_config():\n    return 2'))
    if change == 'motor':
        current.write_text(current.read_text().replace('def model_config():\n    return 1', 'def model_config():\n    return 3'))
    if change == 'patch':
        (new/names[-1]).write_text('changed loader')
    receipt = tmp_path/'summary.json'
    receipt.write_text(json.dumps(dict(completed=change != 'incomplete', source_commit='old-source',
        loader_sha256='changed' if change == 'loader' else 'fixed',
        cases=[dict(name='rl', strict_load=True, auxiliary_tensors=16),
               dict(name='sft', strict_load=True, auxiliary_tensors=0)])))
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *a, **k: 'old-source\n')
    if change == 'environment':
        assert driver.reuse_native_loads(new, old, receipt, 'fixed')['reused']
    else:
        with pytest.raises(ValueError):
            driver.reuse_native_loads(new, old, receipt, 'fixed')
