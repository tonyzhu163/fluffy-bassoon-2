"""Complete fifty-case reward attribution across unequal annotation partitions."""
from copy import deepcopy
import pytest

from fluffy_bassoon.meta_rl.flora_terminal_shards import baseline_episode_shard, gather_baseline_outputs
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol
from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS


def test_fifty_unequal_shards_preserve_grades_and_cannot_release_partial_quality():
    episodes, rows = [], []
    for i in range(50):
        lower = 1+i%4
        episodes.append(dict(id=f'case-{i}', task='Open the drawer.', camera_order=list(REWARD_CAMERAS),
            info_on_video=False, extra_info_on_video=[], raw_rgb_only=True,
            success_termination_disabled=True, horizon=750, completed=True,
            videos=[dict(camera=c, path=f'/qualified/case-{i}-{c}.mp4', sha256='a'*64)
                    for c in REWARD_CAMERAS],
            frame_clock=dict(frame_count=751, native_fps=20., encoded_fps=20., resampled=False,
                physical_steps=750, initial_frame_included=True, timestamp_sha256='b'*64),
            rank=i%8, round=0, local_env=0))
        rows.append(dict(id=f'case-{i}', task='Open the drawer.', score=lower,
            expected_score=lower+.5, progress=(lower-.5)/4,
            probabilities={str(k):.5 if k in (lower,lower+1) else 0. for k in range(1,6)}))
    proto = protocol('OpenDrawer', baseline_count=50)
    shards = {r:list(reversed(rows[r::8])) for r in range(8)}
    assert [len(baseline_episode_shard(episodes,r,proto)) for r in range(8)] == [7,7,6,6,6,6,6,6]
    ordered, labels = gather_baseline_outputs(episodes, shards, proto)
    assert ordered == rows and list(labels) == [e['id'] for e in episodes]
    assert all(labels[row['id']] == (row['expected_score']-1)/4 for row in rows)
    for kind in ('missing_rank','missing_case','duplicate','wrong_rank'):
        bad = deepcopy(shards)
        if kind == 'missing_rank':bad.pop(7)
        elif kind == 'missing_case':bad[7].pop()
        elif kind == 'duplicate':bad[0][-1] = bad[0][0]
        else:bad[0],bad[1] = bad[1],bad[0]
        with pytest.raises(ValueError):gather_baseline_outputs(episodes,bad,proto)
