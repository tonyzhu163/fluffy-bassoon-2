"""Apply a time-varying interface at the boundary of the pinned native motor."""
import numpy as np
from types import SimpleNamespace

from paired_inputs import native
from paired_records import digest, encode, require
from panel import IDENTITY

ev = native()


class Reader:
    def __init__(self, weights):
        params, config = weights
        self.params = params
        self.predict = ev._get_streaming_step_fn(config)
        self.est = ev.StreamingProcrustesEstimator(prior_weight=0.05)

    def update(self, before, step, issued):
        image, position, jaw = before
        next_jaw = float(ev.jaw_width_from_qpos(step["gripper_positions"][-1]))
        command = issued.mean(axis=0)
        ib = image[None,None]
        ia = np.asarray(step["agentview_64"], dtype=np.uint8)[None,None]
        valid = np.ones((1,1), dtype=bool)
        batch = dict(img_before=ib, img_after=ia, dis_flow=ev.compute_dis_flow_maps(ib, ia, valid),
                     issued_mean=command[None,None], valid=valid,
                     delta_eef=(np.asarray(step["positions"][-1], dtype=np.float32)-position)[None,None],
                     delta_jaw=np.array([next_jaw-jaw, next_jaw], dtype=np.float32)[None,None])
        v, w, log_s, gr = self.predict(self.params, batch)
        self.est.update(command[:2], np.asarray(v)[0,0], float(np.asarray(w)[0,0]),
                        log_scale_est=float(np.asarray(log_s)[0]), scale_weight=float(np.linalg.norm(command[:3])+0.05),
                        grip_logit_est=float(np.asarray(gr)[0]), grip_weight=float(abs(command[6])+0.05))


class Episode:
    """Native run_episode handles the query/noise, canonical clipping, rewards and stopping."""
    def __init__(self, sim, policy, arm, row, ref, weights):
        self.sim, self.policy, self.arm, self.row, self.ref = sim, policy, arm, row, ref
        self._model = policy._model
        require((self._model.action_horizon, self._model.action_dim) == (10,32), "Wrong frozen motor shape")
        self.reader = Reader(weights) if arm == "stream11" else None
        self.controls = 0
        self.chunks = []
        self.correction = tuple(row["truth"]) if arm == "exact1" else IDENTITY

    def infer(self, item, *, noise):
        i = len(self.chunks)
        noise_sha = ev.array_digest(noise)
        require(noise_sha == self.ref["noise_draws"][i] and self.controls == i*5, "Actual inference noise/cadence")
        result = self.policy.infer(item, noise=noise)
        proposal = np.asarray(result["actions"], dtype=np.float32)
        self.entry = dict(start=self.controls+1, correction=list(self.correction), noise=noise_sha,
                          proposal=ev.array_digest(proposal))
        if self.controls < 10:
            self.entry["input"] = digest(encode(dict(prompt=item["prompt"],
                        arrays={k:ev.array_digest(v) for k,v in item.items() if k != "prompt"})))
        return result

    def call(self, op, value):
        if op == "create":
            require(self.controls == 0 and not self.chunks and
                    value == dict(self.row["query"], embodiment=self.row["embodiment"]), "Unexpected query create")
            spec, first = self.sim.call(op, value)
            require(spec == dict(value, max_steps=self.ref["reset"]["horizon"]), "Wrong native case")
            self.anchor = np.asarray(first["anchor_eef_pos"], dtype=np.float32)
            self.low = np.asarray(first["action_schema"]["low"], dtype=np.float32)
            self.high = np.asarray(first["action_schema"]["high"], dtype=np.float32)
            self.before = (first["agentview_64"], np.asarray(first["observation"]["robot0_eef_pos"], dtype=np.float32),
                           float(ev.jaw_width_from_qpos(first["observation"]["robot0_gripper_qpos"])))
            return spec, first

        require(op == "chunk" and len(value) == min(5,self.ref["reset"]["horizon"]-self.controls), "Native chunk size")
        issued = value if self.correction == IDENTITY else ev.retarget_action_from_motor(
            value, *self.correction, self.low, self.high)
        issued = np.clip(issued, self.low, self.high).astype(np.float32)
        step = self.sim.call(op, issued)
        n = int(step["steps"])
        self.controls += n
        actual = issued[:n]
        self.entry.update(steps=n, issued=ev.array_digest(actual), canonical=ev.array_digest(value[:n]),
                          changed_offsets=np.flatnonzero(np.any(actual != value[:n], axis=1)).tolist())
        if self.entry["start"] <= 10:
            self.entry["after"] = digest(encode(dict(public=ev.public_digest(step["observation"]),
                                 image=ev.array_digest(step["agentview_64"]), success=bool(step["success"]),
                                 rewards=step["rewards"], terminated=step["terminated"], truncated=step["truncated"],
                                 measured={k:ev.array_digest(step[k]) for k in ("positions","gripper_positions","joint_positions")})))
        self.chunks.append(self.entry)
        done = step["terminated"] or step["truncated"]
        if not done and self.reader is not None and len(self.chunks) <= 24:
            self.reader.update(self.before, step, actual)
            if len(self.chunks) >= 2:
                self.correction = self.reader.est.estimate()
            self.before = (np.asarray(step["agentview_64"], dtype=np.uint8),
                           np.asarray(step["positions"][-1], dtype=np.float32),
                           float(ev.jaw_width_from_qpos(step["gripper_positions"][-1])))
        if self.arm == "exact11" and self.controls >= 10:
            self.correction = tuple(self.row["truth"])
        # Only the motor sees the estimated canonical EEF position. The reader
        # above receives the true worker output in the uncalibrated public frame.
        if self.correction[0] and not done:
            step = dict(step, observation=ev.retarget_observation_for_motor(
                step["observation"], self.anchor, self.correction[0]))
        return step


def run_arm(sim, policy, row, name, ref, resets, weights):
    episode = Episode(sim, policy, name, row, ref, weights)
    rec = ev.run_paired_query(
        ev.run_episode, episode, episode, row["query"], SimpleNamespace(name=row["embodiment"]),
        arm_name=name, resets=resets, key=(row["suite"],row["task_id"],13),
        support_physics=row["support_physics"], expected_prompt=ref["reset"]["prompt"], create_attempts=3)
    # The inner neutral motor reports its fixed correction. The wrapper records the actual correction at each query.
    del rec["inferred"], rec["prefix_xy_disp_15_mm"], rec["prefix_xy_disp_30_mm"]
    rec["chunks"] = episode.chunks
    return rec
