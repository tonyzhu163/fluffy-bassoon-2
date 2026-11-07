"""Qualify horizon-shared joint bounds and explicit source-DEV margin selection.

The original exported per-step bounds and labels stay immutable. This is a
separate candidate fitting profile. TEST tensors never enter this procedure.
"""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor, write_tensor


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
ROOT = Path(__file__).resolve().parent


def qualify():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    profile = os.environ.get('TASK_FIT_BOUNDS_ID', 'decoded-fit-bounds01')
    assert profile in ('decoded-fit-bounds01', 'decoded-fit-bounds02')
    output = BASE/profile
    output.mkdir()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, model_calls=0, optimizer_updates=0, simulator_actions=0,
        test_tensor_loaded=False, source_corpus_admitted=False, profile=profile)
    try:
        source = BASE/'decodedhistory01'
        audit_path = BASE/'decodedhistory01-readback01/RESULT.json'
        audited = json.loads(audit_path.read_bytes())
        exported = json.loads((source/'RESULT.json').read_bytes())
        assert audited['status'] == 'passed' and audited['export_result_sha256'] == sha(source/'RESULT.json')
        assert audited['physical_decisions'] == 5958 and exported['available_teacher_policy_versions'] == [0]

        def read(name):
            binding = exported['files'][name]
            path = source/name
            assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
            return read_tensor(path)

        training = read('train.pt.gz')['trials']
        original = read('train-statistics.pt.gz')
        residual = np.concatenate([row['issued']-row['bases'] for row in training])
        joint = np.abs(residual).max(axis=(0, 1))
        margin, development = 1.1, None
        if profile == 'decoded-fit-bounds02':
            # The finite grid is fixed in Git before this qualification. DEV
            # selects one global hyperparameter; no per-joint DEV statistic
            # enters the envelopes and no TEST tensor is loaded.
            development = read('dev.pt.gz')['trials']
            dev_residual = np.concatenate([row['issued']-row['bases'] for row in development])
            report['precommitted_global_margin_grid'] = [1.5, 2., 4.]
            report['dev_margin_candidates'] = []
            for margin in report['precommitted_global_margin_grid']:
                candidate = np.maximum(joint*np.float32(margin), np.float32(.001))
                outside = int((np.abs(dev_residual) >= candidate).sum())
                report['dev_margin_candidates'].append(dict(margin=margin, coordinates_outside=outside))
                if outside == 0:
                    break
            assert outside == 0, 'No precommitted margin covers source DEV'
        radius = np.broadcast_to(np.maximum(joint*np.float32(margin), np.float32(.001)), (32, 14)).copy()
        reference = (torch.from_numpy(residual).abs().amax(dim=(0, 1))*margin).clamp(min=.001)
        assert torch.equal(torch.from_numpy(radius), reference.expand(32, 14))
        statistics = dict(feature_mean=original['feature_mean'].clone(),
            feature_std=original['feature_std'].clone(), residual_radius=torch.from_numpy(radius),
            fitted_partition='TRAIN feature statistics and joint envelopes',
            radius_rule='max(global_margin*TRAIN max absolute residual over rows AND horizon,0.001) per joint; broadcast H32',
            global_margin=margin, global_margin_selection='source DEV finite-density coverage' if development is not None else 'precommitted constant',
            split_sha256=original['split_sha256'], no_label_clipping=True,
            original_export_statistics_sha256=sha(source/'train-statistics.pt.gz'),
            source_train_sha256=sha(source/'train.pt.gz'), test_tensor_loaded=False)
        path = output/'STATISTICS.pt.gz'
        # Profile01 locks its constant before DEV; profile02 explicitly selects
        # one predeclared global margin on DEV. Both preserve TRAIN envelopes.
        write_tensor(path, statistics)
        report.update(statistics_sha256=sha(path), statistics_locked_before_dev_read=development is None,
            dataset_audit_sha256=sha(audit_path), export_result_sha256=sha(source/'RESULT.json'),
            feature_normalization_unchanged=True, joint_envelopes_fitted_on_train_only=True,
            global_margin=margin, global_margin_selected_on_dev=development is not None,
            radius_per_joint=radius[0].tolist(), coverage={})
        if development is None:
            development = read('dev.pt.gz')['trials']
        for partition, data in [('TRAIN', training), ('DEV', development)]:
            values = torch.from_numpy(np.concatenate([row['issued']-row['bases'] for row in data]))
            scaled = values/torch.from_numpy(radius)
            outside = int((scaled.abs() >= 1).sum())
            report['coverage'][partition] = dict(decisions=len(values), coordinates_outside=outside)
            assert outside == 0, 'This fixed TRAIN-only candidate does not cover source TRAIN/DEV labels'
            latent = torch.atanh(scaled)
            log_jacobian = torch.from_numpy(radius).log()+torch.log1p(-scaled.square())
            assert torch.isfinite(latent).all() and torch.isfinite(log_jacobian).all()
        restored = read_tensor(path)
        for key in ('feature_mean', 'feature_std', 'residual_radius'):
            assert torch.equal(restored[key], statistics[key])
        assert not torch.cuda.is_initialized()
        report.update(status='train_bounds_qualified_on_train_dev', cuda_initialized=False,
            independent_numpy_torch_bound_reference_exact=True,
            inverse_tanh_and_density_jacobian_finite=True,
            original_export_and_statistics_unchanged=True, complete_ad_learning_histories=False,
            student_fit_or_deployment_qualified=False,
            required_next='Bind this common fixed candidate profile in the student fitting contract; qualify training/checkpoint recovery and frozen online controls separately.')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output/'RESULT.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    qualify()
