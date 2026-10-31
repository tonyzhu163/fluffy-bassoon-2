"""Saved-only report denominators and descriptive summaries; no model imports."""
import numpy as np
from fluffy_bassoon.meta_rl.vla_distribution_report import paired,summarize_samples


def test_partial_pairs_use_only_actual_common_queries_and_list_discordant_ids():
    records=[dict(task_id=0,method='native',success=False),dict(task_id=10,method='native',success=True),
        dict(task_id=0,method='ad_full',success=True),dict(task_id=20,method='ad_full',success=True)]
    result=paired(records,'ad_full','native')
    assert result['matched_queries']==1
    assert result['counts']==dict(both_success=0,both_failure=0,left_only=1,right_only=0)
    assert result['task_ids']['left_only']==[0] # Missing cases are not counted as losses.


def test_saved_clipping_summary_is_a_fraction_and_does_not_infer_a_model_mean():
    summary=summarize_samples({'ad_full':{'clipped_coordinate':[np.array([True,False,False,True])],
        'applied_mean':[np.zeros(4,np.float32)],'applied_std':[np.ones(4,np.float32)]}})['ad_full']
    assert summary['clipped_coordinate']['mean']==.5
    assert summary['applied_mean']['mean_abs']==0
    assert summary['applied_std']['mean']==1
