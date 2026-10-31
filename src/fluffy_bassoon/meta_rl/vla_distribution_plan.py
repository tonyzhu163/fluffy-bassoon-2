"""Metadata-only fixed9 mean/dispersion diagnostic, with immutable source500 inputs."""
import argparse
import json
from pathlib import Path
from .vla_source40_noise_plan import build_plan as original_cases, content_hash

FORMAT='source40_distribution_diagnostic_plan_v1'
LEDGER_FORMAT='source40_distribution_diagnostic_budget_v1'
METHODS=('native','ad_full','dpt_full','ad_zero_mean_same_std','dpt_zero_mean_same_std','ad_original_mean_std1','dpt_original_mean_std1')
OPERATIONS={m:dict(mean='zero' if 'zero_mean' in m else 'original',std='unit' if 'std1' in m else 'learned') for m in METHODS if m!='native'}


def build_plan(*,worktree,source_commit,inputs):
    if set(inputs)!={'matrix','matrix_sha256','assembly','assembly_sha256'}:
        raise ValueError('Bind the same immutable source-only final500 matrix/assembly')
    base=original_cases(worktree=worktree,source_commit=source_commit)
    cases=[]
    for entry in base['cases']:
        if entry['case']['task_id']%10:continue
        groups=[dict(group_id=f'{entry["case"]["case_id"]}/{m}/k0',case=entry['case'],method=m,
            source_condition='native' if m=='native' else 'source500',updates=None if m=='native' else 500,
            support_budgets=[0],maximum_actions=410) for m in METHODS]
        cases.append(dict(case=entry['case'],simulator=entry['simulator'],groups=groups))
    value=dict(format=FORMAT,contract=dict(source_commit=source_commit,worktree=str(Path(worktree).resolve()),
        inputs=inputs,operations=OPERATIONS,methods=list(METHODS),maximum_total_actions=27000,maximum_gpu_seconds=3600,
        optimizer_updates=0,qualification_extra_resets=0,source_data_from_targets=False,
        purpose='Diagnostic of learned conditional mean and dispersion; no checkpoint selection or new meta-performance claim',
        native_distribution=dict(kind='independent_native_flow_noise',mean=0.,std=1.,shape=[10,32],
            epsilon_trace_field='native_initial_noise',proposed_flow_noise_trace_field='native_initial_noise',clipping=None)),
        cases=cases,groups=[g for e in cases for g in e['groups']],
        counts=dict(cases=9,physical_episodes=63,nominal_actions_including_settling=25830,action_cap=27000,gpu_seconds=3600))
    value['diagnostic_sha256']=content_hash(value)
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('worktree','matrix','assembly','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-commit','matrix-sha256','assembly-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    value=build_plan(worktree=a.worktree,source_commit=a.source_commit,
        inputs=dict(matrix=str(a.matrix),matrix_sha256=a.matrix_sha256,assembly=str(a.assembly),assembly_sha256=a.assembly_sha256))
    with a.output.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(diagnostic_sha256=value['diagnostic_sha256'],counts=value['counts'])))

if __name__=='__main__':main()
