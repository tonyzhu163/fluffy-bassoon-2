"""Protect executed-action alignment, episode ownership and causal learner inputs.

The stand-in below tests the environment/data boundary, not robotics success.
The stdio test protects actual multimodal MCP interoperability and catches
native/Python stdout corrupting the protocol. Run these on allocated Linux.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from fluffy_bassoon.agent_sim.core import Observation, Session, Transition
from fluffy_bassoon.agent_sim.export import examples, export_episode
from fluffy_bassoon.agent_sim.robocasa import RoboCasaBackend


class CounterBackend:
    """A deliberately synthetic fixture; never exposed as a production backend."""
    def __init__(self):
        self.n = 0
        self.fail_at = None
        self.end_at = None

    def describe(self):
        return dict(benchmark="test-only", action_dim=2, state_order=["z", "a"])

    def observation(self):
        return Observation({"camera": np.full((3, 4, 3), self.n, np.uint8)},
                           {"z": np.array([self.n]), "a": np.array([-self.n])},
                           "test instruction", self.n * .05)

    def reset(self, task, seed):
        self.n = 0
        # Must not corrupt a stdio client even when simulator code writes fd 1.
        print("fixture Python diagnostic", flush=True)
        os.write(1, b"fixture native diagnostic\n")
        return self.observation()

    def canonicalize(self, actions):
        if actions.shape[1] != 2:
            raise ValueError("Two coordinates required")
        actions[:, 1] = actions[:, 1] >= .5
        return actions

    def step(self, action):
        self.n += 1
        if self.n == self.fail_at:
            raise RuntimeError("physical dispatch failed")
        return Transition(self.observation(), action, float(self.n), self.n == self.end_at,
                          False, self.n == 1, self.snapshot())

    def snapshot(self):
        return dict(secret_object_pose=[987654321], step=self.n)

    def close(self):
        pass


def make_session(tmp_path, feedback="hidden", max_controls=4):
    return Session(CounterBackend(), tmp_path / "collection", client="codex", model="test",
                   source_commit="test", max_controls=max_controls, max_chunk=3, feedback=feedback)


def move(session, actions, control=0, summary="before-action-summary"):
    return session.execute(actions, observation_control=control, subgoal="approach",
                           summary=summary, expected_outcome="visible movement")


def test_control_rate_alignment_and_causal_export(tmp_path):
    s = make_session(tmp_path)
    s.reset("Task", 7, "episode1")
    result = move(s, [[.2, .7], [.3, .1]])
    assert result["executed_actions"] == [[.2, 1.], [.3, 0.]]
    assert "success_once" not in result["status"]
    assert "last_reward" not in result
    move(s, [[.4, .5]], control=2, summary="second-decision-only")
    s.finish(summary="final-outcome-only", success_claim=False)
    ep = s.output / "episode1"
    rows = list(examples(ep))
    assert len(rows) == 2
    assert rows[0]["input"]["history"] == []
    assert rows[0]["target"]["requested_actions"][0][1] == .7
    assert rows[0]["target"]["executed_actions"][0][1] == 1.
    assert rows[0]["outcome"]["rewards"] == [1., 2.]
    first = json.dumps(rows[0]["input"])
    assert "before-action-summary" not in first
    second = json.dumps(rows[1]["input"])
    assert "before-action-summary" in second
    assert "second-decision-only" not in second
    assert "final-outcome-only" not in second
    assert "987654321" not in json.dumps(rows)
    assert "reward" not in second and "success_once" not in second
    export_episode(ep, tmp_path / "export")
    with np.load(tmp_path / "export/trajectory.npz", allow_pickle=False) as tape:
        np.testing.assert_array_equal(tape["states"], [[0, 0], [1, -1], [2, -2], [3, -3]])
        np.testing.assert_array_equal(tape["decision_ids"], [0, 0, 1])
        assert tape["actions"].shape == (3, 2)
        assert tape["sim_times"].shape == (4,)
    assert len(list((ep / "observations").glob("*/camera.png"))) == 4
    assert not json.loads((ep / "result.json").read_text())["agent_success_claim"]
    assert json.loads((ep / "result.json").read_text())["native_success_once"]
    s.close()


def test_native_feedback_only_enters_later_inputs(tmp_path):
    s = make_session(tmp_path, feedback="native")
    s.reset("Task", 7, "episode1")
    move(s, [[.2, 0.]])
    move(s, [[.3, 1.]], control=1)
    s.finish(summary="done", success_claim=True)
    rows = list(examples(s.output / "episode1"))
    assert "last_reward" not in json.dumps(rows[0]["input"])
    assert rows[1]["input"]["history"][0]["result"]["last_reward"] == 1.
    assert list(examples(s.output / "episode1", history_decisions=0))[1]["input"]["history"] == []
    s.close()


def test_behavior_conditioning_does_not_contaminate_planner_inputs(tmp_path):
    """Protect the distinction between hindsight descriptors and online context."""
    s = make_session(tmp_path)
    s.reset("Task", 1, "episode1")
    move(s, [[.1, 0.], [.2, 0.]])
    move(s, [[.3, 1.]], control=2)
    s.finish(summary="done", success_claim=True)
    labels = dict(format="agent_sim_annotations_v1", episode_id="episode1", quality=2,
        source=dict(kind="human", identity="reviewer", rubric="Task-specific execution quality rubric v1"),
        segments=[dict(start_control=0, end_control=3, subtask="annotated grasp",
                       mistake=True, achieved=False)])
    export_episode(s.output / "episode1", tmp_path / "annotated", annotations=labels)
    planner = (tmp_path / "annotated/examples.jsonl").read_text()
    assert "annotated grasp" not in planner and "reviewer" not in planner
    policy = [json.loads(x) for x in (tmp_path / "annotated/policy_examples.jsonl").read_text().splitlines()]
    row = policy[0]
    assert row["input"]["conditioning"]["episode_metadata"]["quality"] == 2
    assert row["input"]["conditioning"]["episode_metadata"]["episode_controls"] == 3
    assert row["input"]["conditioning"]["mistake"] is True
    assert row["annotation_source"]["kind"] == "human"
    assert row["future_image_supervision"]["control"] == 3
    assert row["future_image_supervision"]["achieved_subgoal"] is False
    assert "000003" not in json.dumps(row["input"])
    # A segment boundary inside a committed chunk would mislabel its actions.
    labels["segments"][0]["end_control"] = 1
    with pytest.raises(ValueError, match="decision boundaries"):
        export_episode(s.output / "episode1", tmp_path / "invalid", annotations=labels)
    assert not (tmp_path / "invalid").exists()
    export_episode(s.output / "episode1", tmp_path / "unannotated")
    row = json.loads((tmp_path / "unannotated/policy_examples.jsonl").read_text().splitlines()[0])
    assert row["input"]["conditioning"]["episode_metadata"]["quality"] is None
    assert row["future_image_supervision"]["achieved_subgoal"] is None
    s.close()


def test_reject_invalid_and_stale_actions_before_dispatch(tmp_path):
    s = make_session(tmp_path)
    s.reset("Task", 1, "episode1")
    for actions in ([], [[float("nan"), 0]], [[0]], [[0, 0]] * 4):
        with pytest.raises(ValueError):
            move(s, actions)
    assert s.backend.n == 0
    move(s, [[0, 0]])
    with pytest.raises(ValueError, match="Stale"):
        move(s, [[0, 0]])
    with pytest.raises(RuntimeError, match="Finish"):
        s.reset("Task", 1, "episode2")
    assert s.backend.n == 1
    s.finish(summary="stop", success_claim=False)
    with pytest.raises(FileExistsError):
        s.reset("Task", 1, "episode1")
    s.close()


def test_terminal_prefix_and_budget_do_not_create_unexecuted_labels(tmp_path):
    s = make_session(tmp_path)
    s.backend.end_at = 1
    s.reset("Task", 1, "episode1")
    result = move(s, [[0, 0]] * 3)
    assert result["executed_controls"] == 1
    with pytest.raises(RuntimeError):
        move(s, [[0, 0]], control=1)
    s.finish(summary="ended", success_claim=False)
    row = next(examples(s.output / "episode1"))
    assert len(row["target"]["executed_actions"]) == 1
    assert len(row["target"]["requested_actions"]) == 3
    s.close()


def test_uncertain_dispatch_cannot_resume_or_export(tmp_path):
    s = make_session(tmp_path)
    s.backend.fail_at = 2
    s.reset("Task", 1, "episode1")
    with pytest.raises(RuntimeError, match="dispatch failed"):
        move(s, [[0, 0]] * 3)
    with pytest.raises(RuntimeError, match="failed"):
        move(s, [[0, 0]], control=1)
    assert s.backend.n == 2
    assert not (s.output / "episode1/result.json").exists()
    with pytest.raises(FileNotFoundError):
        list(examples(s.output / "episode1"))
    s.close()


def test_robocasa_canonical_commands_preserve_native_thresholds():
    backend = RoboCasaBackend()
    actions = np.zeros((3, 12))
    actions[:, 6] = [-1, .49, .5]
    actions[:, 11] = [.5, .49, 1]
    actions[:, 0] = [-1, .1, 1]
    actual = backend.canonicalize(actions.copy())
    np.testing.assert_array_equal(actual[:, 6], [0, 0, 1])
    np.testing.assert_array_equal(actual[:, 11], [1, 0, 1])
    np.testing.assert_array_equal(actual[:, 0], actions[:, 0])
    with pytest.raises(ValueError, match="silently clip"):
        backend.canonicalize(np.full((1, 12), 2.))


def test_real_mcp_stdio_images_and_persistent_episode(tmp_path):
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        root = Path(__file__).resolve().parents[1]
        parameters = StdioServerParameters(command=sys.executable, args=[
            "-m", "fluffy_bassoon.agent_sim", "--output", str(tmp_path / "mcp"),
            "--client", "claude", "--model", "test", "--source-commit", "test",
            "--max-controls", "3", "--backend-factory", "test_agent_sim:CounterBackend"],
            env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])})
        with anyio.fail_after(30):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    assert len((await client.list_tools()).tools) == 5
                    reset = await client.call_tool("sim_reset", dict(task="Task", seed=1, episode_id="episode1"))
                    assert not reset.isError
                    assert any(block.type == "image" for block in reset.content)
                    response = await client.call_tool("sim_execute", dict(actions=[[.1, .8]],
                        observation_control=0, subgoal="move", summary="visible target", expected_outcome="motion"))
                    assert not response.isError, response
                    assert json.loads(response.content[0].text)["executed_actions"] == [[.1, 1.]]
                    stale = await client.call_tool("sim_execute", dict(actions=[[.1, .8]],
                        observation_control=0, subgoal="move", summary="stale", expected_outcome="motion"))
                    assert stale.isError
                    end = await client.call_tool("sim_finish", dict(summary="done", success_claim=False))
                    assert not end.isError
    anyio.run(run)
