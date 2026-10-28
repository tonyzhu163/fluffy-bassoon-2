"""Per-observation meta-RL interface to the repository's single CMM implementation.

The recurrence lives in lerobot_policy_ctm.architectures.continuous_memory_machine
(arXiv:2610.07907, section 3.2 and appendix A). This wrapper keeps the compact
trainer's keyword API and returns synchronization features, not a readout.
"""

from lerobot_policy_ctm.architectures.continuous_memory_machine.core import (
    CMMConfig, ContinuousMemoryMachine, JointMemoryBlock)


MemoryBlock = JointMemoryBlock


class CMMCore(ContinuousMemoryMachine):
    """`forward(observation[B,I], state)` -> (features[B,width], complete state)."""

    def __init__(self, input_dim, *, width=64, stm_length=10, ltm_tokens=32,
                 layers=1, heads=4, synapse_depth=8, ticks=1, dropout=0., gated=False, pair_seed=1000):
        super().__init__(input_dim, None, CMMConfig(
            width=width, stm_length=stm_length, ltm_slots=ltm_tokens, memory_layers=layers,
            synapse_depth=synapse_depth, heads=heads, ticks=ticks, dropout=dropout,
            learned_residual_gate=gated, pair_seed=pair_seed))

    def forward(self, observation, state=None):
        return self.step(observation, state)
