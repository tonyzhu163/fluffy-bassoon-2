"""Native RoboCasa terminal videos and graded reward numerical contract."""
import pytest
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import baseline_validation, flora_jobs, parse_outputs
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol


def test_native_video_and_probability_grade_use_actual_horizon():
    episode = dict(id="native-1", task="open the drawer", horizon=750, completed=True,
        camera_order=["robot0_agentview_left"], raw_rgb_only=True, info_on_video=False,
        extra_info_on_video=[], success_termination_disabled=True, rank=0, round=0, local_env=0,
        video=dict(path="/native/episode.mp4", sha256="a" * 64), frame_clock=dict(
            frame_count=751, native_fps=25., encoded_fps=25., physical_steps=750,
            resampled=False, initial_frame_included=True, timestamp_sha256="b" * 64))
    p = protocol("OpenDrawer")
    assert flora_jobs([episode], p) == [dict(id="native-1", task="open the drawer", videos=["/native/episode.mp4"])]
    row = dict(id="native-1", task="open the drawer", expected_score=2.25, progress=.3125,
        probabilities={"1": 0., "2": .75, "3": .25, "4": 0., "5": 0.})
    assert parse_outputs([episode], [row], p) == {"native-1": .3125}
    episode["simulator_final_success"] = True
    with pytest.raises(ValueError):
        flora_jobs([episode], p)


def test_eight_native_cases_require_headroom_and_both_score_classes():
    rows = [dict(id=str(i), expected_score=4. if i < 3 else 2., simulator_final_success=i < 3) for i in range(8)]
    actual = baseline_validation(rows, protocol("OpenDrawer"))
    assert actual["go"] and actual["AUROC"] == 1. and actual["initial_success_rate"] == 3 / 8
    for row in rows:
        row["simulator_final_success"] = True
    actual = baseline_validation(rows, protocol("OpenDrawer"))
    assert not actual["go"] and actual["AUROC"] is None
