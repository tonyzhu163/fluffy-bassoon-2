"""Shared CTM state transition; conditioning is supplied by each architecture.

Two synchronization options, both off by default so saved experts are unchanged:
`bounded_synchronisation` reads alpha/beta (a decayed mean) followed by a
parameter-free RMS norm instead of alpha/sqrt(beta), whose scale grows with the
ticks accumulated; `decay_per_observation` spreads each learned decay over the
ticks of one observation, so the memory horizon in observations no longer
shrinks as ticks are added.
"""
import torch
from torch.nn import functional as F
from .token_ctm import TokenCTM


class PersistentCTMCore(TokenCTM):
    def _synchronisation_decays(self):
        with torch.no_grad():
            self.decay_params_action.clamp_(0, 15)
            self.decay_params_out.clamp_(0, 15)
        if getattr(self, "decay_per_observation", False):
            return ((self.decay_params_action / -self.num_ticks).exp(),
                    (self.decay_params_out / -self.num_ticks).exp())
        return self.decay_params_action.neg().exp(), self.decay_params_out.neg().exp()

    def _normalise_synchronisation(self, alpha, beta):
        if getattr(self, "bounded_synchronisation", False):
            return F.rms_norm(alpha / beta, alpha.shape[-1:], eps=1e-8)
        return alpha / torch.sqrt(beta)

    def _initial_activated_state(self, rows):
        return self.start_activated_state.expand(rows, -1)

    def _synchronisation_memory_shapes(self, rows):
        return {key: (rows, 2 * self.hidden_dim) for key in
                ("action_alpha", "action_beta", "out_alpha", "out_beta")}

    def _initial_synchronisation_memory(self, activated, r_out):
        alpha, beta = self._synchronisation_accumulators(activated, None, None, r_out, "out")
        return dict(action_alpha=None, action_beta=None, out_alpha=alpha, out_beta=beta)

    def _synchronisation_accumulators(self, activated, alpha, beta, decay, kind):
        """Update random-pair accumulators without computing a discarded readout.

        Other pairing schemes retain the reference implementation. Parameter
        names, state layout and arithmetic order of the carried memory agree
        with compute_synchronisation; normalization remains at the read sites.
        """
        if self.neuron_select_type != "random-pairing":
            _, alpha, beta = self.compute_synchronisation(activated, alpha, beta, decay, kind)
            return alpha, beta
        left = getattr(self, f"{kind}_neuron_indices_left")
        right = getattr(self, f"{kind}_neuron_indices_right")
        product = activated[:, left] * activated[:, right]
        if alpha is None or beta is None:
            return product, torch.ones_like(product)
        return decay * alpha + product, decay * beta + 1

    def _attention_synchronisation(self, activated, memory, r_action):
        alpha, beta = self._synchronisation_accumulators(
            activated, memory["action_alpha"], memory["action_beta"], r_action, "action")
        return self._normalise_synchronisation(alpha, beta), {**memory, "action_alpha": alpha, "action_beta": beta}

    def _advance_post_activation(self, activated, memory, r_out):
        alpha, beta = self._synchronisation_accumulators(
            activated, memory["out_alpha"], memory["out_beta"], r_out, "out")
        return {**memory, "out_alpha": alpha, "out_beta": beta}

    def _output_synchronisation(self, memory):
        return self._normalise_synchronisation(memory["out_alpha"], memory["out_beta"])

    def validate_memory(self, memory, batch_size):
        rows, width = batch_size * self.population_count, self.hidden_dim
        shapes = {"activated": (rows, width), "trace": (rows, width, self.history_length),
                  **self._synchronisation_memory_shapes(rows)}
        if not isinstance(memory, dict) or set(memory) != set(shapes):
            raise ValueError("Require the complete CTM memory, or None at an episode start")
        for key, shape in shapes.items():
            tensor = memory[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                    or tensor.device != self.start_trace.device or tensor.dtype != self.start_trace.dtype):
                raise ValueError(f"Invalid CTM memory tensor: {key}")

    def decode_persistent(self, context_mask, projected, conditioning, memory):
        if (context_mask.ndim != 2 or not context_mask.shape[0] or context_mask.dtype != torch.bool
                or context_mask.device != self.start_trace.device or len(projected) != 2
                or any(value.shape != (*context_mask.shape, self.hidden_dim)
                       or value.dtype != self.start_trace.dtype or value.device != context_mask.device
                       for value in projected)):
            raise ValueError("Require projected K/V [B,P,D] and boolean mask [B,P] on the expert device")
        batch, slots, width = context_mask.shape[0], self.population_count, self.hidden_dim
        if conditioning.shape != (batch, slots, width):
            raise ValueError("Conditioning must contain one token per CTM population")
        keys, values = projected
        valid = torch.cat((context_mask, context_mask.new_ones(batch, slots)), dim=1)
        r_action, r_out = self._synchronisation_decays()
        if memory is None:
            activated = self._initial_activated_state(batch * slots)
            trace = self.start_trace.expand(batch * slots, -1, -1)
            sync_memory = self._initial_synchronisation_memory(activated, r_out)
        else:
            self.validate_memory(memory, batch)
            activated, trace = memory["activated"], memory["trace"]
            noise_std = float(getattr(self, "memory_noise_std", 0.0))
            if self.training and noise_std > 0:
                # Perturb the carried neuron state and its pre-activation history relative to
                # their own spread; synchronisation accumulators are left intact.
                activated = activated + torch.randn_like(activated) * activated.detach().std() * noise_std
                trace = trace + torch.randn_like(trace) * trace.detach().std() * noise_std
            sync_memory = {key: memory[key] for key in self._synchronisation_memory_shapes(batch * slots)}
        # Incoming memory remains immutable for independent branches and checkpoint replay.
        for _ in range(self.num_ticks):
            synch, sync_memory = self._attention_synchronisation(activated, sync_memory, r_action)
            query = self.q_proj(synch).view(batch, slots, width)
            slot_tokens = self.slot_norm(conditioning + activated.view(batch, slots, width))
            slot_keys, slot_values = self.attention.project_kv(slot_tokens)
            attended = self.attention(query, torch.cat((keys, slot_keys), dim=1),
                                      torch.cat((values, slot_values), dim=1), valid)
            attended = attended.reshape(batch * slots, width) + conditioning.reshape(batch * slots, width)
            preactivation = self.synapses(torch.cat((attended, activated), dim=-1))
            trace = torch.cat((trace[:, :, 1:], preactivation.unsqueeze(-1)), dim=-1)
            activated = self.trace_processor(trace)
            sync_memory = self._advance_post_activation(activated, sync_memory, r_out)
        synch = self._output_synchronisation(sync_memory)
        memory = dict(activated=activated, trace=trace, **sync_memory)
        return self.velocity_proj(synch).view(batch, slots, self.readout_dim), memory
