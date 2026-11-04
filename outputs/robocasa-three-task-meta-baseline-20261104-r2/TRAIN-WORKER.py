#!/usr/bin/env python3
"""Verify the parent's assigned GPU before running one existing learner."""
import runpy
import sys

import torch
from fluffy_bassoon.baige_gpu import verify_selected_cuda

verify_selected_cuda(torch)
module = sys.argv.pop(1)
if module not in (
    "fluffy_bassoon.meta_rl.vla_history_train",
    "fluffy_bassoon.meta_rl.history_value_train",
):
    raise ValueError("Unknown learner")
runpy.run_module(module, run_name="__main__")
