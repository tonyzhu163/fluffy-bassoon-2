"""Offline behavior conditioning, distinct from the causal planner transcript.

Inspired by pi0.7: describe mixed-quality behavior at training time and supply
desired behavior at deployment. Retrospective metadata is NOT an online oracle.
"""
from copy import deepcopy


def validate_annotations(value, episode_id, controls, boundaries):
    if value is None:
        return None
    if (value.get("format") != "agent_sim_annotations_v1"
            or value.get("episode_id") != episode_id):
        raise ValueError("Annotations must name this episode and their format")
    source = value.get("source", {})
    if (source.get("kind") not in ("human", "model")
            or not all(isinstance(source.get(k), str) and source[k].strip()
                       for k in ("identity", "rubric"))):
        raise ValueError("Annotations require human/model identity and a quality rubric")
    quality = value.get("quality")
    if quality is not None and (type(quality) is not int or not 1 <= quality <= 5):
        raise ValueError("Quality is an annotated 1–5 score or null, never inferred from success")
    previous = 0
    for segment in value.get("segments", []):
        start, end = segment.get("start_control"), segment.get("end_control")
        if (type(start) is not int or type(end) is not int or not previous <= start < end <= controls
                or start not in boundaries or end not in boundaries):
            raise ValueError("Segments must be ordered, disjoint and align to decision boundaries")
        if not isinstance(segment.get("subtask"), str) or not segment["subtask"].strip():
            raise ValueError("Every annotated segment needs a subtask")
        for key in ("mistake", "achieved"):
            if segment.get(key) is not None and type(segment[key]) is not bool:
                raise ValueError("Mistake and achieved annotations must be bool or null")
        previous = end
    return deepcopy(value)


def policy_examples(rows, observations, result, annotations=None):
    """Create action supervision conditioned on explicit behavior descriptors.

    Future images are a separately marked prediction target, not policy input.
    An annotated successful segment endpoint can supervise a goal generator.
    Unannotated chunk endpoints are only observed futures, not achieved goals.
    """
    boundaries = {row["input"]["observation"]["control"] for row in rows}
    boundaries.add(result["controls"])
    annotations = validate_annotations(annotations, rows[0]["episode_id"],
                                       result["controls"], boundaries)
    metadata = dict(episode_controls=result["controls"],
        episode_sim_seconds=observations[-1]["sim_time"] - observations[0]["sim_time"],
        quality=None if annotations is None else annotations.get("quality"))
    for row in rows:
        start = row["input"]["observation"]["control"]
        end = start + len(row["target"]["executed_actions"])
        segment = next((s for s in (annotations or {}).get("segments", [])
                        if s["start_control"] <= start and end <= s["end_control"]), None)
        context = row["target"]["context"]
        interface = row["input"]["interface"]
        conditioning = dict(subtask=context["subgoal"] if segment is None else segment["subtask"],
            episode_metadata=metadata, mistake=None if segment is None else segment.get("mistake"),
            control_modality=interface.get("action_format", "backend_declared"))
        goal_control = end if segment is None else segment["end_control"]
        yield dict(format="agent_sim_conditioned_policy_example_v1",
            episode_id=row["episode_id"], decision=row["decision"],
            input=dict(observation=deepcopy(row["input"]["observation"]),
                history=deepcopy(row["input"]["history"]), interface=deepcopy(interface),
                conditioning=deepcopy(conditioning)),
            target=dict(executed_actions=deepcopy(row["target"]["executed_actions"])),
            future_image_supervision=dict(control=goal_control,
                images=deepcopy(observations[goal_control]["images"]),
                achieved_subgoal=None if segment is None else segment.get("achieved"),
                source="observed_chunk_endpoint" if segment is None else "annotated_segment_endpoint",
                deployment="Predict or externally supply goals; never read a future rollout frame"),
            outcome=deepcopy(row["outcome"]), provenance=deepcopy(row["provenance"]),
            annotation_source=None if annotations is None else deepcopy(annotations["source"]),
            conditioning_source="retrospective_training_descriptors",
            deployment="Supply desired quality/duration/mistake descriptors and a generated or coached "
                       "subtask; do not use realized future episode statistics at evaluation")
