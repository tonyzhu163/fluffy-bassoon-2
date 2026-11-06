"""Fail closed before this run dispatches any candidate into a robot readout."""
import hashlib
import json
import math
from pathlib import Path

FORMAT = 'corrected_steering_readout_admission_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assess(effect, thresholds):
    failures = []
    for name, minimum in thresholds.items():
        metric = name.removeprefix('minimum_')
        value = effect.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Require finite measured steering metrics')
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError('Require finite nonnegative prospective thresholds')
        if value < minimum:
            failures.append(f'{metric} {value:.8g} < {minimum:.8g}')
    return dict(passed=not failures, failures=failures, thresholds=thresholds)


def binding(fit, run_contract_path):
    fit = Path(fit)
    contract = json.loads((fit / 'contract.json').read_text())
    run = json.loads(Path(run_contract_path).read_text())
    if (contract.get('stopping_rule') != 'relative_abs_reference_v2'
            or contract.get('run_contract_sha256') != sha(run_contract_path)
            or contract.get('interface') != 'kv_bridge' or contract.get('objective') != 'ad'
            or contract.get('assembly_sha256') != run['assembly_sha256']
            or contract.get('native_model_sha256') != run['native_model_sha256']
            or contract.get('native_norm_sha256') != run['native_norm_sha256']):
        raise ValueError('Require this run\'s corrected primary KV-AD fit contract')
    return dict(candidate_sha256=sha(fit / 'candidate.pt'),
        fit_contract_sha256=sha(fit / 'contract.json'), run_contract_sha256=sha(run_contract_path),
        validation_sha256=contract['validation_sha256'], native_model_sha256=contract['native_model_sha256'],
        native_norm_sha256=contract['native_norm_sha256'], source_commit=contract['source_commit'])


def publish(fit, run_contract_path, effect, source_commit):
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _json
    fit = Path(fit)
    status = json.loads((fit / 'training-status.json').read_text())
    run = json.loads(Path(run_contract_path).read_text())
    gate = assess(effect, run['steering_gate'])
    result = dict(format=FORMAT, binding=binding(fit, run_contract_path), effect=effect, gate=gate,
        corrected_validation_convergence=status.get('converged') is True
            and status.get('stop_reason') == 'validation_plateau'
            and status.get('stopping_rule') == 'relative_abs_reference_v2',
        source_commit=source_commit, selected_update=status['selected_update'], optimizer_updates_during_measurement=0)
    result['admitted'] = result['corrected_validation_convergence'] and gate['passed']
    _json(fit / 'steering-admission.json', result)
    return result


def require(fit, run_contract_path):
    fit = Path(fit)
    result = json.loads((fit / 'steering-admission.json').read_text())
    run = json.loads(Path(run_contract_path).read_text())
    status = json.loads((fit / 'training-status.json').read_text())
    if (result.get('format') != FORMAT or result.get('binding') != binding(fit, run_contract_path)
            or result.get('admitted') is not True or result.get('corrected_validation_convergence') is not True
            or status.get('converged') is not True or status.get('stop_reason') != 'validation_plateau'
            or status.get('stopping_rule') != 'relative_abs_reference_v2'
            or result.get('source_commit') != result['binding']['source_commit']
            or result.get('selected_update') != status.get('selected_update')
            or result.get('optimizer_updates_during_measurement') != 0
            or not assess(result['effect'], run['steering_gate'])['passed']
            or result.get('gate') != assess(result['effect'], run['steering_gate'])):
        raise ValueError('Candidate has no matching, converged, passing prospective steering admission')
    return result
