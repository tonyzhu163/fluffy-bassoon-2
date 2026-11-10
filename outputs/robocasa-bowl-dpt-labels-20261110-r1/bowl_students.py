"""The same two LoRA banks as the PPO pilot, with a causal transformer for bowl histories."""
import torch

from fit import PairedStudents
from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore


def terminal_features(encoder, history):
    tokens = history.transpose(0, 1)
    valid = tokens[..., -1].bool()
    features = encoder.sequence(tokens[..., :-1], reset=torch.zeros_like(valid), valid=valid)
    last = valid.long().sum(0) - 1
    return features[last, torch.arange(len(last), device=last.device)]


class BowlStudents(PairedStudents):
    def __init__(self, native, history_width):
        super().__init__(native, history_width)
        self.encoder = CausalDecoderCore(history_width, width=128, memory_length=102, layers=2, heads=4)

    def select(self, name, history):
        self.active = name
        self.context = None
        if name == "dpt":
            self.context = self.project(terminal_features(self.encoder, history))
