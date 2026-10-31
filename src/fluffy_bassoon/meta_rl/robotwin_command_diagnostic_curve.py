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


def validate_resource(ticket,clock,*,operation,plan,other_fit_tickets=()):
    if operation not in ('cache04','fit'):raise ValueError('Only declared diagnostic operations')
    family='fixed12-command-diagnostic/'
    limit=plan['stage_gpu_limits'][operation]
    if (plan.get('diagnostic_purpose')!=bank.PURPOSE or plan.get('source_eligible') is not False
            or plan.get('admitted') is not True or type(limit) is not int or not 120<=limit<=clock['worker_seconds']
            or plan['clock_contract']!=clock['contract'] or plan['clock_root']!=ticket['clock_root']
            or not plan.get('prior_cost_receipts') or not plan.get('storage_admission')):
        raise ValueError('Actual finite linked/original plan with old costs and storage proof required')
    known={family+'cache04',family+'fit-action',family+'fit-kv',family+'readout'}
    if any(r.get('lineage','').startswith(family) and r['lineage'] not in known for r in clock['attempts']):
        raise ValueError('Unknown/stale diagnostic lineage cannot disappear from stage accounting')
    if operation=='cache04':
        if ticket.get('kind')!='cache04':raise ValueError('Explicit cache04 kind required')
        caps={family+'cache04':limit};lineage=family+'cache04'
    else:
        heads=plan['fit_head_gpu_limits'];kind=ticket.get('kind')
        if (kind not in ('action','kv') or set(heads)!={'action','kv'}
                or any(type(v) is not int or not 120<=v<=clock['worker_seconds'] for v in heads.values())
                or sum(heads.values())>limit):
            raise ValueError('Exact immutable per-head caps must fit the aggregate fit-stage ceiling')
        caps={family+'fit-'+k:v for k,v in heads.items()};lineage=family+'fit-'+kind
    rows=[r for r in clock['attempts'] if r.get('lineage') in caps]
    current=[r for r in rows if r['path']==ticket['resource_attempt'] and r['lineage']==lineage]
    if (len(current)!=1 or any(r.get('lineage_seconds')!=caps[r['lineage']] for r in rows)
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in rows)>limit):
        raise ValueError('Both heads/all actual and open reservations must fit the original stage ceiling')
    for name,cap in caps.items():
        related=[r for r in rows if r['lineage']==name]
        if sum(r.get('charged_seconds',r['reserved_seconds']) for r in related)>cap:
            raise ValueError('No cross-head borrowing or per-attempt cap reset')
    if operation=='fit':
        if plan.get('fit_topology')!='separate_one_gpu_physical_allocations':
            raise ValueError('Per-head tickets do not support double-booking one grouped physical allocation')
        tickets=[ticket,*other_fit_tickets];by_attempt={};hosts=set()
        for item in tickets:
            attempt=item['resource_attempt'];host=item['hostname']
            if attempt in by_attempt or host in hosts or item.get('kind') not in ('action','kv') or item.get('gpu_count')!=1:
                raise ValueError('Each physical allocation must have exactly one head booking/ticket')
            by_attempt[attempt]=item;hosts.add(host)
        if set(by_attempt)!={r['path'] for r in rows}:
            raise ValueError('Bind all other existing fit tickets; no omitted physical allocations')
        for row in rows:
            item=by_attempt[row['path']]
            if item.get('operation')!='fit' or family+'fit-'+item['kind']!=row['lineage']:
                raise ValueError('Actual physical ticket kind differs from the booked head')
    if any(r['lineage']==lineage and r['path']!=ticket['resource_attempt'] and r['status']=='started' for r in rows):
        raise ValueError('Only the other head may have a simultaneous registered reservation')


def fit_curve(kind,backend,bank_value,cache_receipts,wide_configuration,output,*,guard,deadline,contract,resume=None):
    return curve.fit_curve(kind,backend,bank_value,cache_receipts,wide_configuration,output,
        guard=guard,deadline=deadline,contract=contract,resume=resume,
        _api=bank,_profile=PROFILE,_validate=validate_profile)


def main(argv=None,*,operation='fit'):
    return curve.main(argv,_api=bank,_profile=PROFILE,_validate=validate_profile,
        _core_bank_sha=CORE_BANK_SHA,_operation=operation,_validate_resource=validate_resource)


if __name__=='__main__':raise SystemExit(main())
