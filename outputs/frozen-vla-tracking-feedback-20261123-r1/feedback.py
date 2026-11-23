"""Causal five-control motor feedback from issued actions and public wrist displacements."""
import numpy as np


class Feedback:
    def __init__(self, contract, lag):
        if lag not in (1, 3):
            raise ValueError("Expected a most recent or a three-chunk-old residual")
        self.lag = lag
        self.prior = np.asarray(contract["prior_matrix"])
        self.bias = np.asarray(contract["source_fit"]["command_bias"])
        self.rho = float(contract["source_fit"]["rho"])
        body = self.prior[:3]
        self.inverse = np.linalg.solve(body @ body.T + contract["source_residual_power"]*np.eye(3),body).T
        self.actions = []
        self.innovations = []

    def correct(self, proposal):
        issued = np.array(proposal, dtype=np.float32, copy=True)
        if len(self.innovations) < 3:
            return issued
        correction = self.rho * self.innovations[-self.lag] @ self.inverse
        norm = np.linalg.norm(correction)
        # Bound each applied delta by the pending translation norm. Holding pose stays a hold.
        if norm:
            factor = np.minimum(1., np.linalg.norm(issued[:, :3],axis=1)/norm)
            issued[:, :3] = np.clip(issued[:, :3] + factor[:,None]*correction, -1., 1.)
        return issued

    def observe(self, commands, before, after):
        if len(commands) != 5:
            return  # Native episodes stop on success; never feed a partial chunk into the five-control law.
        current = np.mean(commands[:, :3],axis=0,dtype=np.float64)
        previous = self.actions[-1] if self.actions else np.zeros(3)
        older = self.actions[-2] if len(self.actions)>1 else np.zeros(3)
        measured = np.asarray(after,dtype=np.float64) - before
        residual = np.r_[current,previous] @ self.prior - measured
        innovation = residual - np.r_[1.,current,previous,older] @ self.bias
        self.actions.append(current)
        self.innovations.append(innovation)


class FeedbackTrace:
    def __init__(self, trace, feedback):
        self.trace = trace
        self.feedback = feedback
        self.corrections = []

    def call(self, op, value):
        if op == "create":
            if self.corrections:
                raise ValueError("Use a new feedback state for every episode")
            return self.trace.call(op,value)
        issued = self.feedback.correct(value)
        before = self.trace.obs["robot0_eef_pos"].copy()
        step = self.trace.call(op,issued)
        n = step["steps"]
        self.feedback.observe(issued[:n], before, step["observation"]["robot0_eef_pos"])
        self.corrections.append((issued[:n]-value[:n]).tolist())
        return step
