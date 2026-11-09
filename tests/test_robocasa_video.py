"""Camera association, orientation and incomplete-stream exclusion."""
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.robocasa.video import EpisodeVideos, REWARD_CAMERAS, observation_frames, raw_frames


def test_raw_and_public_observations_produce_identical_upright_camera_streams(tmp_path):
    frames=[np.arange(18,dtype=np.uint8).reshape(2,3,3)+offset for offset in (0,30,60)]
    raw={camera+'_image':frame[::-1] for camera,frame in zip(REWARD_CAMERAS,frames)}
    public=dict(main_images=frames[0][None],wrist_images=frames[1][None],
                extra_view_images=frames[2][None,None])
    for a,b in zip(raw_frames(raw),observation_frames(public)):
        np.testing.assert_array_equal(a,b)
    writers=[]
    class Writer:
        def __init__(self,path,**kwargs):
            self.path,self.frames=Path(path),[];writers.append(self)
        def append_data(self,frame):self.frames.append(frame.copy())
        def close(self):self.path.write_bytes(b'encoded-stand-in')
    recorder=EpisodeVideos(tmp_path,'episode',fps=20,writer_factory=Writer)
    recorder.append(raw_frames(raw))
    recorder.append([frame+1 for frame in observation_frames(public)])
    recorder.close()
    assert recorder.count==2 and [ref['camera'] for ref in recorder.refs()]==list(REWARD_CAMERAS)
    for writer,frame in zip(writers,frames):
        np.testing.assert_array_equal(writer.frames[0],frame)
        np.testing.assert_array_equal(writer.frames[1],frame+1)
    del public['wrist_images']
    with pytest.raises(ValueError,match='all required'):observation_frames(public)


def test_partial_multicamera_write_closes_every_stream_and_cannot_be_scored(tmp_path):
    writers=[]
    class Writer:
        def __init__(self,path,**kwargs):self.closed=False;self.fail=False;writers.append(self)
        def append_data(self,frame):
            if self.fail:raise OSError('encoder failed')
        def close(self):self.closed=True
    recorder=EpisodeVideos(tmp_path,'episode',fps=20,writer_factory=Writer)
    frames=[np.zeros((2,3,3),np.uint8)]*3
    recorder.append(frames)
    writers[1].fail=True
    with pytest.raises(OSError,match='encoder failed'):recorder.append(frames)
    assert all(writer.closed for writer in writers)
    with pytest.raises(ValueError):recorder.refs()
