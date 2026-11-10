"""The same two LoRA banks as the PPO pilot, with a causal transformer for bowl histories."""
import torch

from fit import PairedStudents
from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore


class BowlStudents(PairedStudents):
    def __init__(self, native, history_width):
        super().__init__(native, history_width)
        self.encoder = CausalDecoderCore(history_width, width=128, memory_length=102, layers=2, heads=4)

    def select(self, name, history):
        self.active = name
        self.context = None
        if name == "dpt":
            tokens = history.transpose(0, 1)
            valid = tokens[..., -1].bool()
            features = self.encoder.sequence(tokens[..., :-1], reset=torch.zeros_like(valid), valid=valid)
            terminal = valid.long().sum(0) - 1
            self.context = self.project(features[terminal, torch.arange(len(terminal),device=terminal.device)])
