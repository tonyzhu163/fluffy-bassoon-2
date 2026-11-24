"""Example adapters for harness.Hooks. Select with HOOKS=adapters:Name."""
from harness import Hooks

# Task string in the public LIBERO release (and the SmolVLA training data) for libero_goal/3.
# LIBERO-Pro2.0 BASE words it "open the top layer of the drawer and put the bowl inside".
LIBERO_GOAL = {"open the top layer of the drawer and put the bowl inside": "open the top drawer and put the bowl inside"}


class LiberoPrompt(Hooks):
    """Instruction as worded in the original LIBERO release, for the whole episode."""
    name = "libero-prompt"

    def prompt(self, prompt, obs, context):
        return LIBERO_GOAL.get(prompt, prompt)


class Replan10(Hooks):
    """Consume ten controls per chunk (LeRobot's LIBERO evaluation example) instead of all 50."""
    name = "replan10"
    replan = 10


class LiberoPromptReplan10(LiberoPrompt):
    name = "libero-prompt-replan10"
    replan = 10
