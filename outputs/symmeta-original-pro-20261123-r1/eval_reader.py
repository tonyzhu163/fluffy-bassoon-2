"""The seed-41 Stage 2A online estimator, with per-update measurements."""
import numpy as np

from eval_stage2a_live_ur5e import (
    StreamingProcrustesEstimator, _get_streaming_step_fn, compute_dis_flow_maps,
    jaw_width_from_qpos, load_v2_checkpoint,
)
from fluffy_bassoon.xemb.embodiment import downsample_agentview


IDENTITY = (0., 1., 1)
UPDATE_LIMIT = 24


class OnlineReader:
    def __init__(self, model):
        self.params, cfg, _ = model
        self.step_fn = _get_streaming_step_fn(cfg)
        self.est = StreamingProcrustesEstimator(prior_weight=0.05)
        self.adapter = IDENTITY

    def observe(self, before, after, issued):
        # This method receives only observed images, proprioception and the commands actually sent.
        ib, ia = [downsample_agentview(o["agentview_image"])[None, None] for o in (before, after)]
        pos0, pos1 = [np.asarray(o["robot0_eef_pos"], np.float32) for o in (before, after)]
        jaw0, jaw1 = [float(jaw_width_from_qpos(o["robot0_gripper_qpos"])) for o in (before, after)]
        mean = issued.mean(axis=0)
        valid = np.ones((1, 1), dtype=bool)
        batch = dict(img_before=ib, img_after=ia, dis_flow=compute_dis_flow_maps(ib, ia, valid),
                     issued_mean=mean[None, None], delta_eef=(pos1-pos0)[None, None],
                     delta_jaw=np.array([jaw1-jaw0, jaw1], np.float32)[None, None], valid=valid)
        vec, weight, ls, grip = (np.asarray(x) for x in self.step_fn(self.params, batch))
        weight = float(weight[0, 0])
        xy = float(np.linalg.norm(mean[:2]))
        self.est.update(mean[:2], vec[0, 0], weight, log_scale_est=float(ls[0]),
                        scale_weight=float(np.linalg.norm(mean[:3]) + 0.05),
                        grip_logit_est=float(grip[0]), grip_weight=float(abs(mean[6]) + 0.05))
        if self.est.n_chunks >= 2:
            self.adapter = self.est.estimate()
        return dict(chunk=self.est.n_chunks, issued_mean=mean.tolist(), issued_xy_norm=xy,
                    visual_xy=vec[0, 0].tolist(), visual_weight=weight, log_scale=float(ls[0]),
                    grip_logit=float(grip[0]), eligible_yaw=bool(xy >= self.est.min_excitation and weight > 0),
                    accumulated_visual_weight=self.est.weight_sum, dot=self.est.dot_sum,
                    cross=self.est.cross_sum, next_adapter=self.adapter)
