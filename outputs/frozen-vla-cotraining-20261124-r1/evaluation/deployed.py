"""Run all non-opening joint history interventions on the same deployed sampler computation."""
import numpy as np


class Deployed:
    def __init__(self, motor):
        self.motor = motor
        self.native = getattr(motor, "_native", None)

    def digests(self):
        return self.motor.digests()

    def infer(self, item, *, noise, condition, history):
        return self.motor.infer(item, noise=noise, condition=condition, history=history)

    def history_infer(self, item, *, noise, condition, history):
        if self.native is None:
            # The original injected engine runs its single computation for all history values.
            return self.infer(item, noise=noise, condition=condition, history=history)
        import jax
        import jax.numpy as jnp
        from cotrain_policy import transformed

        # Use the recovered trainer's compiled nonempty-history sampler even for masked history.
        # This holds the numerical implementation constant in fresh/zero/lagged causal contrasts.
        native = self.native
        obs, _ = transformed(self.motor.stock, item)
        offset = native.offset(native.heads[condition], jnp.asarray(history)[None])
        actions = native.with_history(native.states[condition], jax.random.key(0), obs,
                                      noise=jnp.asarray(noise)[None], offset=offset)
        return native.decode(obs, actions)

    def parity(self, item, *, noise):
        result = self.motor.parity(item, noise=noise)
        if self.native is not None:
            from cotrain_policy import array_sha
            # Same installed history graph as the aligned/zero/lagged trials, with the original
            # kernel and identically zero branch. Log the result separately from native plain-kernel parity.
            actions = self.history_infer(item, noise=noise, condition="original", history=np.zeros(52, np.float32))["actions"]
            stock = self.motor.stock.infer(item, noise=noise)["actions"]
            result.update(history_graph_noop_sha256=array_sha(actions), repeated_stock_sha256=array_sha(stock),
                          history_graph_noop_max_abs=float(np.abs(actions-stock).max()),
                          history_graph_noop_issued_prefix_max_abs=float(np.abs(actions[:5]-stock[:5]).max()))
        return result
