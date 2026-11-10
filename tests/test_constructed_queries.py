import json
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.constructed_queries import ConstructedQueryStudent, export_query, measured_label, query_batch


def test_branched_outcomes_change_only_the_query_target(tmp_path):
    candidates = ['neutral', 'left', 'right']; digest = 'b' * 64
    public = tmp_path/'public.npz'
    np.savez(public, trial_id='a', decision_sha256=digest, candidate_ids=candidates,
             support_obs=np.array([[1,2],[3,4]],np.float32), delivered_actions=np.array([[.5],[-.5]],np.float32),
             native_rewards=np.array([0,1],np.float32), query_obs=np.array([5,6],np.float32))
    rows = [dict(trial_id='a', decision_sha256=digest, restore_verified=True, candidate=c,
                 native_success=c=='right', signed_miss_m=0.01) for c in reversed(candidates)]
    paths=[]
    for i in range(2):
        evidence=tmp_path/'evidence.json'; evidence.write_text(json.dumps(rows)); dest=tmp_path/f'q{i}.npz'
        export_query(public,evidence,dest); paths.append(dest)
        for r in rows: r['native_success']=r['candidate']=='left'
    tokens,valid,targets=query_batch(paths,candidates)
    assert targets.tolist()==[2,1]; torch.testing.assert_close(tokens[:,0],tokens[:,1])
    torch.testing.assert_close(tokens[-1,0],torch.tensor([5.,6.,-.5,1.,1.,1.]))
    student=ConstructedQueryStudent(tokens.shape[-1],candidates,width=8,layers=1,heads=2)
    loss=student.query_loss(paths); loss.backward()
    assert loss.isfinite() and student.readout.weight.grad.abs().sum()>0
    for r in rows: r['native_success']=True
    assert measured_label(candidates,rows,trial_id='a',decision_sha256=digest) is None
    for r in rows: r['native_success']=False
    assert measured_label(candidates,rows,trial_id='a',decision_sha256=digest) is None
    rows[0]['trial_id']='other'
    with pytest.raises(ValueError,match='match'):
        measured_label(candidates,rows,trial_id='a',decision_sha256=digest)
