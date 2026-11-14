"""Synchronized RGB streams for the shared RoboCasa terminal reward prompt."""
from pathlib import Path

from .flora_ppo import sha

# Match the policy's left / wrist / right layout in the official collage.
REWARD_CAMERAS = ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right")


def observation_frames(observation):
    """Native public observations already contain upright RGB for all views."""
    import numpy as np
    try:
        return [np.asarray(observation["main_images"])[0],
                np.asarray(observation["wrist_images"])[0],
                np.asarray(observation["extra_view_images"])[0, 0]]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Left, wrist and right RGB are all required for reward recording") from error


def raw_frames(observation):
    """Native OpenGL observations need the same vertical flip as the policy."""
    try:
        return [observation[camera + "_image"][::-1] for camera in REWARD_CAMERAS]
    except (KeyError, TypeError) as error:
        raise ValueError("Left, wrist and right raw RGB are all required") from error


class EpisodeVideos:
    """Append every camera at each physical tick, including reset and final."""
    def __init__(self, root, key, *, fps, writer_factory):
        self.paths = [Path(root) / f"{key}-{camera}.mp4" for camera in REWARD_CAMERAS]
        if any(path.exists() for path in self.paths):
            raise ValueError("Never overwrite native episode video evidence")
        self.writers, self.shapes = [], None
        self.count, self.closed, self.failed = 0, False, False
        try:
            for path in self.paths:
                self.writers.append(writer_factory(str(path), fps=fps, codec="libx264",
                                                  quality=8, macro_block_size=None))
        except BaseException:
            self.close()
            raise

    def append(self, frames):
        import numpy as np
        if self.closed or len(frames) != len(REWARD_CAMERAS):
            raise ValueError("Exactly three synchronized camera frames are required")
        frames = [np.asarray(frame) for frame in frames]
        if any(frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 for frame in frames):
            raise ValueError("Each camera must supply native HWC uint8 RGB")
        shapes = [frame.shape for frame in frames]
        if self.shapes is not None and self.shapes != shapes:
            raise ValueError("Camera image dimensions cannot change during an episode")
        self.shapes = shapes
        try:
            for writer, frame in zip(self.writers, frames):
                writer.append_data(np.ascontiguousarray(frame))
        except BaseException:
            self.failed = True
            self.close()
            raise
        self.count += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        error = None
        for writer in self.writers:
            try:
                writer.close()
            except BaseException as failure:
                self.failed = True
                if error is None:
                    error = failure
        if error is not None:
            raise error

    def refs(self):
        if not self.closed or not self.count or self.failed:
            raise ValueError("Close the complete synchronized videos before scoring")
        return [dict(camera=camera, path=str(path), sha256=sha(path))
                for camera, path in zip(REWARD_CAMERAS, self.paths)]


class EpisodeFrameDigests:
    """Native-reward evaluation can retain frame hashes without encoded videos.

    These records are explicitly not playable videos or rewarder inputs. Every
    observed RGB frame contributes to a per-camera digest and the frame clock.
    """
    def __init__(self, root, key):
        import hashlib
        self.path = Path(root) / f'{key}-frame-digests.json'
        if self.path.exists():
            raise ValueError('Never overwrite frame evidence')
        self.digests = [hashlib.sha256() for _ in REWARD_CAMERAS]
        self.count, self.shapes, self.closed = 0, None, False

    def append(self, frames):
        import numpy as np
        frames = [np.asarray(frame) for frame in frames]
        if (self.closed or len(frames) != len(REWARD_CAMERAS)
                or any(frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 for frame in frames)):
            raise ValueError('Exactly three native uint8 RGB frames are required')
        shapes = [list(frame.shape) for frame in frames]
        if self.shapes is not None and shapes != self.shapes:
            raise ValueError('Camera image dimensions cannot change')
        self.shapes = shapes
        for digest, frame in zip(self.digests, frames, strict=True):
            digest.update(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self):
        if not self.closed:
            import json
            self.path.write_text(json.dumps(dict(format='native_rgb_frame_digests_v1',
                playable_video=False, frame_count=self.count, shapes=self.shapes,
                camera_order=list(REWARD_CAMERAS), sha256=[d.hexdigest() for d in self.digests]),
                indent=2) + '\n')
            self.closed = True

    def refs(self):
        if not self.closed or not self.count:
            raise ValueError('Close the frame record before scoring')
        return [dict(path=str(self.path), sha256=sha(self.path))]
