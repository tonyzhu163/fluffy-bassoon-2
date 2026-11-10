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


def test_native_branch_targets_preserve_delivered_controls_and_select_by_id(tmp_path):
    from fluffy_bassoon.meta_rl.constructed_queries import native_branch_chunks

    candidates = ["left", "right"]
    digest = "1" * 64
    stats = {"actions": {"mean": [.1] * 32, "std": [2.] * 32}}
    actions = np.zeros((57, 12), np.float32)
    actions[:, 1] = np.arange(57) / 100
    actions[:, 6] = 1.
    actions[:, 11] = -1.
    targets, masks = native_branch_chunks(actions, [0, 50, 56], stats)
    assert targets.shape == (3, 50, 32) and masks.dtype == np.bool_
    np.testing.assert_array_equal(masks.sum(1), [50, 7, 1])
    for i, start in enumerate((0, 50, 56)):
        n = min(50, len(actions) - start)
        assert masks[i, :n].all() and not masks[i, n:].any()
        recovered = targets[i, :n] * (2. + 1e-6) + .1
        np.testing.assert_allclose(recovered[:, :12], actions[start:start+n], atol=3e-7)
        np.testing.assert_allclose(recovered[:, 12:], 0., atol=3e-7)

    public = tmp_path / "public.npz"
    np.savez(public, trial_id="trial", decision_sha256=digest, candidate_ids=candidates,
             support_obs=np.zeros((1, 2), np.float32), delivered_actions=np.ones((1, 1), np.float32),
             native_rewards=np.zeros(1, np.float32), query_obs=np.zeros(2, np.float32))
    rows = [dict(candidate=c, trial_id="trial", restore_verified=True, decision_sha256=digest,
                 native_success=c == "right", executed_actions=actions[-7:].tolist())
            for c in reversed(candidates)]
    rows[1].pop("executed_actions")
    evidence = tmp_path / "branches.json"
    evidence.write_text(json.dumps(rows))
    dest = tmp_path / "target.npz"
    export_query(public, evidence, dest, action_stats=stats)
    with np.load(dest) as example:
        tokens = example["tokens"].copy()
        assert example["query_label"] == 1
        np.testing.assert_array_equal(example["query_flow_target"], targets[1])
        np.testing.assert_array_equal(example["query_delivered"], masks[1])
    rows[1]["native_success"] = True
    assert measured_label(candidates, rows, trial_id="trial", decision_sha256=digest) is None
    rows[1]["signed_miss_m"] = 0.
    assert measured_label(candidates, rows, trial_id="trial", decision_sha256=digest) is None
    rows[1]["native_success"] = False
    rows[0]["native_success"] = False
    evidence.write_text(json.dumps(rows))
    empty = tmp_path / "abstention.npz"
    export_query(public, evidence, empty, action_stats=stats)
    with np.load(empty) as abstained:
        assert abstained["query_label"] == -1 and not abstained["query_delivered"].any()
        np.testing.assert_array_equal(tokens, abstained["tokens"])
    with pytest.raises(ValueError, match="offsets"):
        native_branch_chunks(actions, [57], stats)
    with pytest.raises(ValueError, match="binary"):
        native_branch_chunks(np.zeros((1, 12), np.float32), [0], stats)
