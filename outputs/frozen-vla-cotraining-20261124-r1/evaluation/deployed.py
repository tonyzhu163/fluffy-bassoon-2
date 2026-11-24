"""Compose the locked native projection with its trained history offset."""
import numpy as np


class Deployed:
    def __init__(self, motor):
        self.motor = motor
        self.native = getattr(motor, "_native", None)

    def digests(self):
        return self.motor.digests()

    def infer(self, item, *, noise, condition, history):
        return self.motor.infer(item, noise=noise, condition=condition, history=history)

    def _sample_offset(self, item, noise, condition, history, *, force_add=False):
        import jax
        import jax.numpy as jnp
        from cotrain_policy import transformed

        native = self.native
        obs, _ = transformed(self.motor.stock, item)
        args = native.states[condition], jax.random.key(0), obs
        noise = jnp.asarray(noise)[None]
        offset = native.offset(native.heads[condition], jnp.asarray(history)[None])
        # On the 4090 even an identically zero addition can compile to a different
        # denoiser. The identity element uses the untouched native Euler sampler with
        # this condition's materialized kernel. Decide from the computed offset.
        if force_add or np.any(np.asarray(offset)):
            actions = native.with_history(*args, noise=noise, offset=offset)
        else:
            actions = native.plain(*args, noise=noise)
        return native.decode(obs, actions)

    def history_infer(self, item, *, noise, condition, history):
        if self.native is None:
            return self.infer(item, noise=noise, condition=condition, history=history)
        return self._sample_offset(item, noise, condition, history)

    def forced_zero(self, item, noise, condition):
        """Unissued diagnostic of the compiler effect at fixed observation, kernel, noise and zero offset."""
        return self._sample_offset(item, noise, condition, np.zeros(52, np.float32), force_add=True)["actions"]

    def parity(self, item, *, noise):
        result = self.motor.parity(item, noise=noise)
        if self.native is not None:
            from cotrain_policy import array_sha

            # The untrained history head yields an exactly zero offset even with *valid*
            # history. Exercise the same dispatcher used by all trained joint controls.
            history = np.tile(np.array([0.0]*11 + [0.04, 1.0], np.float32), 4)
            routed = self.history_infer(item, noise=noise, condition="original", history=history)["actions"]
            stock = self.motor.stock.infer(item, noise=noise)["actions"]
            result.update(deployed_noop_sha256=array_sha(routed), repeated_stock_sha256=array_sha(stock),
                          deployed_noop_max_abs=float(np.abs(routed-stock).max()))
            forced = self.forced_zero(item, noise, "original")
            result.update(forced_history_graph_sha256=array_sha(forced),
                          forced_history_graph_max_abs=float(np.abs(forced-stock).max()),
                          forced_history_graph_issued_prefix_max_abs=float(np.abs(forced[:5]-stock[:5]).max()))
        return result
