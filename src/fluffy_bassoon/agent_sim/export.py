"""Causal JSONL examples and numeric action tapes from completed episodes.

Inputs contain only observations and earlier *visible* tool results. Current
decision text/actions are targets. Native outcomes are separate labels; private
simulator state never enters an input. No action is mislabeled as motor noise.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import FORMAT, encode, write_json


def read_episode(path):
    path = Path(path).resolve()
    session = json.loads((path.parent / "session.json").read_text())
    episode = json.loads((path / "episode.json").read_text())
    result = json.loads((path / "result.json").read_text())
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    if (session["format"] != FORMAT or episode["format"] != FORMAT or not result["completed"]
            or not events or events[-1]["kind"] != "finish"
            or any(e["kind"] in ("error", "interrupted") for e in events)):
        raise ValueError("Only complete, unambiguous episodes are eligible for export")
    return session, episode, result, events


def examples(path, *, history_decisions=None):
    if history_decisions is not None and (type(history_decisions) is not int or history_decisions < 0):
        raise ValueError("History length must be nonnegative, or None for full history")
    session, episode, _, events = read_episode(path)
    history = []
    interface = next(e["interface"] for e in events if e["kind"] == "reset")
    pending = None
    transitions = []
    for event in events:
        if event["kind"] == "decision":
            if pending is not None:
                raise ValueError("Decision without a completed tool result")
            pending = event
            transitions = []
        elif event["kind"] == "transition":
            if pending is None or event["decision"] != pending["decision"]:
                raise ValueError("Transition has no matching decision")
            transitions.append(event)
        elif event["kind"] == "tool_result":
            if (pending is None or not transitions or event["decision"] != pending["decision"]
                    or len(transitions) != event["result"]["executed_controls"]):
                raise ValueError("Incomplete physical action trace")
            prior = history if history_decisions is None else history[-history_decisions:] if history_decisions else []
            # Copy through JSON so a caller cannot mutate a later example's history.
            inputs = json.loads(encode(dict(observation=pending["observation"],
                history=prior, interface=interface, feedback=session["feedback"])))
            target = dict(context=pending["context"], requested_actions=pending["requested_actions"],
                          executed_actions=[t["action"] for t in transitions])
            yield dict(format="agent_sim_training_example_v1",
                episode_id=episode["episode_id"], decision=pending["decision"],
                input=inputs, target=target,
                outcome=dict(rewards=[t["reward"] for t in transitions],
                    success=[t["success"] for t in transitions],
                    terminated=[t["terminated"] for t in transitions],
                    truncated=[t["truncated"] for t in transitions]),
                provenance=dict(task=episode["task"], seed=episode["seed"],
                    client=session["client"], model=session["model"],
                    source_commit=session["source_commit"]))
            history.append(dict(context=pending["context"], requested_actions=pending["requested_actions"],
                                observation=pending["observation"], result=event["result"]))
            pending = None
    if pending is not None:
        raise ValueError("Unfinished decision")


def export_episode(path, output, *, history_decisions=None, annotations=None):
    from .conditioning import policy_examples
    path, output = Path(path).resolve(), Path(output)
    session, episode, result, events = read_episode(path)
    rows = list(examples(path, history_decisions=history_decisions))
    transitions = [e for e in events if e["kind"] == "transition"]
    observations = [e["observation"] for e in events if e["kind"] == "reset"]
    observations.extend(e["next_observation"] for e in transitions)
    if (not transitions or len(observations) != len(transitions) + 1
            or result["controls"] != len(transitions) or result["decisions"] != len(rows)
            or [o["control"] for o in observations] != list(range(len(observations)))):
        raise ValueError("Require a complete consecutive control-rate episode")
    policy_rows = list(policy_examples(rows, observations, result, annotations))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "examples.jsonl").open("x") as stream:
        for row in rows:
            stream.write(encode(row) + "\n")
    with (output / "policy_examples.jsonl").open("x") as stream:
        for row in policy_rows:
            stream.write(encode(row) + "\n")
    if annotations is not None:
        write_json(output / "annotations.json", annotations)
    interface = next(e["interface"] for e in events if e["kind"] == "reset")
    state_order = interface.get("state_order", list(observations[0]["state"]))
    states = np.asarray([[v for k in state_order for v in o["state"][k]] for o in observations], np.float32)
    with (output / "trajectory.npz").open("xb") as stream:
        np.savez_compressed(stream,
            states=states, actions=np.asarray([t["action"] for t in transitions], np.float32),
            rewards=np.asarray([t["reward"] for t in transitions], np.float32),
            terminated=np.asarray([t["terminated"] for t in transitions], bool),
            truncated=np.asarray([t["truncated"] for t in transitions], bool),
            success=np.asarray([t["success"] for t in transitions], bool),
            decision_ids=np.asarray([t["decision"] for t in transitions], np.int64),
            sim_times=np.asarray([o["sim_time"] for o in observations], np.float64))
    write_json(output / "manifest.json", dict(format="agent_sim_training_export_v1",
        session_root=str(path.parent), episode=episode, source=session, state_order=state_order,
        images=[o["images"] for o in observations], controls=len(transitions), decisions=len(rows),
        history_decisions=history_decisions,
        policy_view="policy_examples.jsonl uses retrospective behavior conditioning, "
                    "requiring desired descriptors at deployment. Future images are labels only.",
        train_note="input is causal; target and outcome are labels. Images are relative to session_root. "
                   "Actions use the recorded native interface, never frozen-VLA noise coordinates. "
                   "Split by task/scene/session before making support/query examples."))
    return dict(controls=len(transitions), decisions=len(rows), output=str(output.resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-decisions", type=int)
    parser.add_argument("--annotations", type=Path, help="Optional sourced offline quality/segment labels")
    args = parser.parse_args(argv)
    annotations = None if args.annotations is None else json.loads(args.annotations.read_text())
    print(encode(export_episode(args.episode, args.output,
        history_decisions=args.history_decisions, annotations=annotations)))


if __name__ == "__main__":
    main()
