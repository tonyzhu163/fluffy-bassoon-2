from lerobot_policy_ctm.architectures.recurrent_transformer_control.action_expert import RecurrentTransformerExpert


class PersistentTransformerExpert(RecurrentTransformerExpert):
    def memory_input(self, memory, batch):
        return (self.initial_memory.expand(batch, -1, -1)
                if memory is None else memory["tokens"])
