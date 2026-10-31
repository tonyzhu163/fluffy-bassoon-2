"""Explicit all12 command-diagnostic gate; common5000 curve/ABI unchanged."""
from copy import deepcopy
from . import robotwin_reactive_curve as curve
from . import robotwin_command_diagnostic_bank as bank

FORMAT='robotwin_fixed12_command_diagnostic_curve_v1'
CORE_BANK_SHA='91de73d421537fa01715da12a2aecfc19bdbf1a5edb47054d9274590c7792506'
PROFILE=dict(deepcopy(curve.PROFILE),format=FORMAT,diagnostic_purpose=bank.PURPOSE,
    source_eligible=False,successful_teacher_bank_eligible=False,
    diagnostic_bank_format=bank.FORMAT,complete_cache_format=bank.COMPLETE_FORMAT,
    failed04_label_source='original be247 event-teacher-phone-04; no later recipe substitution',
    maximum_cache04_output_bytes=2<<30)


def validate_profile(profile):
    if profile!=PROFILE:raise ValueError('Explicit all12 command diagnostic profile required; old21eb unchanged')


def validate_resource(ticket,clock,*,operation,plan):
    lineage='fixed12-command-diagnostic/'+('cache04' if operation=='cache04' else 'fits')
    limit=plan['stage_gpu_limits'][operation]
    if (plan.get('diagnostic_purpose')!=bank.PURPOSE or plan.get('source_eligible') is not False
            or plan.get('admitted') is not True or type(limit) is not int or not 120<=limit<8*3600
            or plan['clock_contract']!=clock['contract'] or plan['clock_root']!=ticket['clock_root']
            or not plan.get('prior_cost_receipts') or not plan.get('storage_admission')):
        raise ValueError('Actual finite linked/original plan with old costs and storage proof required')
    rows=[r for r in clock['attempts'] if r.get('lineage')==lineage]
    current=[r for r in rows if r['path']==ticket['resource_attempt']]
    if (len(current)!=1 or any(r.get('lineage_seconds')!=limit for r in rows)
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in rows)>limit):
        raise ValueError('Existing-owner aggregate diagnostic lineage must include all prior attempts')


def fit_curve(kind,backend,bank_value,cache_receipts,wide_configuration,output,*,guard,deadline,contract,resume=None):
    return curve.fit_curve(kind,backend,bank_value,cache_receipts,wide_configuration,output,
        guard=guard,deadline=deadline,contract=contract,resume=resume,
        _api=bank,_profile=PROFILE,_validate=validate_profile)


def main(argv=None,*,operation='fit'):
    return curve.main(argv,_api=bank,_profile=PROFILE,_validate=validate_profile,
        _core_bank_sha=CORE_BANK_SHA,_operation=operation,_validate_resource=validate_resource)


if __name__=='__main__':raise SystemExit(main())
