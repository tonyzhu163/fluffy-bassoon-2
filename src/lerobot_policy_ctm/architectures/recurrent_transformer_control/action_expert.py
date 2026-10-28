from lerobot_policy_ctm.shared.transformer_expert import TransformerExpert


class RecurrentTransformerExpert(TransformerExpert):
    def __init__(self, config):
        super().__init__(config)
        self.block = self.make_block(config)

    def blocks(self):
        return (self.block,) * self.depth
