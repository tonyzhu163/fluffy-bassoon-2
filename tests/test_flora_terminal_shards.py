"""Reward-boundary core: fixed groups, unchanged grades and all64 labels before GAE."""
from copy import deepcopy

import pytest

from fluffy_bassoon.meta_rl.flora_terminal_shards import gather_terminal_outputs, shard_episode_bank
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import parse_outputs
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol
from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS


def bank():
    episodes, outputs = [], []
    for index in range(64):
        key, lower = f'episode-{index:03d}', 1 + index % 4
        episodes.append(dict(id=key, task='open the drawer', camera_order=list(REWARD_CAMERAS),
            info_on_video=False, extra_info_on_video=[], raw_rgb_only=True,
            success_termination_disabled=True, horizon=600, completed=True,
            videos=[dict(camera=camera, path=f'/qualified/{key}-{camera}.mp4', sha256='a' * 64)
                    for camera in REWARD_CAMERAS],
            frame_clock=dict(frame_count=601, native_fps=20., encoded_fps=20., resampled=False,
                physical_steps=600, initial_frame_included=True, timestamp_sha256='b' * 64),
            rank=index // 8, round=index % 8, local_env=0))
        expected = lower + .5
        outputs.append(dict(id=key, task='open the drawer', score=lower, expected_score=expected,
            progress=(expected - 1) / 4, probabilities={str(k): .5 if k in (lower, lower + 1) else 0.
                for k in range(1, 6)}, output=str(lower), rationale=None,
            duration_s=37.55, sampled_frames=64))
    return episodes, outputs


@pytest.mark.parametrize('baseline_count', [8, 50])
def test_eight_terminal_shards_preserve_all64_graded_rewards_and_chronology(baseline_count):
    episodes, rows = bank()
    proto = protocol('OpenDrawer', baseline_count=baseline_count)
    shards = {rank: list(reversed(rows[rank * 8:(rank + 1) * 8])) for rank in reversed(range(8))}
    for rank in range(8):
        assert [row['id'] for row in shard_episode_bank(episodes, rank, 8, proto)] == [
            row['id'] for row in episodes[rank * 8:(rank + 1) * 8]]
    ordered, labels = gather_terminal_outputs(episodes, shards, proto)
    assert ordered == rows and labels == parse_outputs(episodes, rows, proto)
    assert all(value in (.125, .375, .625, .875) for value in labels.values())
    assert all(row['expected_score'] == rows[index]['expected_score']
               and row['probabilities'] == rows[index]['probabilities'] for index, row in enumerate(ordered))


def test_missing_duplicate_misrouted_or_malformed_shard_cannot_release_partial_rewards():
    episodes, rows = bank()
    proto = protocol('OpenDrawer')
    complete = {rank: rows[rank * 8:(rank + 1) * 8] for rank in range(8)}
    for kind in ('missing_rank', 'missing_label', 'duplicate', 'wrong_rank', 'bad_probability'):
        shards = deepcopy(complete)
        if kind == 'missing_rank':
            shards.pop(7)
        elif kind == 'missing_label':
            shards[7].pop()
        elif kind == 'duplicate':
            shards[7][-1] = deepcopy(shards[7][0])
        elif kind == 'wrong_rank':
            shards[0], shards[1] = shards[1], shards[0]
        else:
            shards[0][0]['probabilities']['1'] = .1
        before = deepcopy(shards)
        with pytest.raises(ValueError):
            gather_terminal_outputs(episodes, shards, proto)
        assert shards == before
