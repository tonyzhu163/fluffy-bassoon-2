"""Numerical paired-AUROC qualification for the prospectively sized native bank."""
import pytest

from fluffy_bassoon.robocasa.flora_ppo import baseline_validation
from fluffy_bassoon.robocasa.ppo import protocol


def fifty_rows():
    # Ten positives at4 beat all35 negatives. Five positives at3 beat15
    # negatives at2 and tie20 at3: (350+75+50)/(15*35)=19/21.
    return [dict(id=f'case-{i}', expected_score=float(score), simulator_final_success=truth)
            for i,(score,truth) in enumerate(
                [(4,True)]*10 + [(3,True)]*5 + [(3,False)]*20 + [(2,False)]*15)]


def test_fifty_case_auroc_matches_fixed_reference_with_ties():
    result = baseline_validation(fifty_rows(), protocol('OpenDrawer', baseline_count=50))
    assert result['count'] == 50 and result['initial_success_rate'] == .3
    assert result['AUROC'] == pytest.approx(19/21) and result['go']
    assert result['protocol']['baseline_count'] == 50
    assert result['policy_optimizer_updates'] == 0
    assert 'threshold' not in result and 'cutoff' not in result


def test_fifty_bank_cannot_accept_partial_duplicate_or_reinterpreted_eight_cases():
    rows = fifty_rows()
    expected = protocol('OpenDrawer', baseline_count=50)
    for wrong in (rows[:8], rows[:-1], rows + [dict(rows[0], id='extra')], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            baseline_validation(wrong, expected)
    eight = [dict(id=f'original-{i}', expected_score=4. if i<4 else 2.,
                  simulator_final_success=i<4) for i in range(8)]
    original = baseline_validation(eight, protocol('OpenDrawer'))
    assert original['count'] == 8 and original['AUROC'] == 1 and original['go']


def test_fifty_single_class_retains_undefined_auroc_and_failed_gate():
    for success in (False, True):
        rows = [dict(id=f'case-{i}', expected_score=3., simulator_final_success=success)
                for i in range(50)]
        result = baseline_validation(rows, protocol('OpenDrawer', baseline_count=50))
        assert result['AUROC'] is None and not result['go']
        assert result['count'] == 50 and result['initial_success_rate'] == int(success)
