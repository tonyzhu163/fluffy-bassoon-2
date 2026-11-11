"""Read completed native motor receipts without importing the model runtime."""
import argparse
import json
from pathlib import Path


def summarize(folder):
    result = json.loads((folder / "motor/result.json").read_text())
    clock = json.loads((folder / "clock.json").read_text())
    construction = json.loads((folder / "motor/construction-receipt.json").read_text())
    probe = json.loads((folder / "gpu-probe.json").read_text())
    assert clock["exit_code"] == 0 and result["complete"]
    assert probe["status"] == "passed" and result["paired_initial_observations"]
    assert construction["strict_checkpoint_load"]
    assert construction["exact_saved_tensors_verified"] == 967
    count = result["n_test_per_candidate"]
    candidates = result["results"]
    assert len(candidates) == 2
    initial = [candidate["reset_hashes"][:count] for candidate in candidates]
    assert all(len(hashes) == count for hashes in initial)
    assert initial[0] == initial[1]
    assert len({json.dumps(hashes, sort_keys=True) for hashes in initial[0]}) == count
    rows = []
    for candidate in candidates:
        scores = {int(key.rsplit("_", 1)[1]): value
                  for key, value in candidate["scores"].items()
                  if key.startswith("score/test/")}
        assert set(scores) == set(range(count))
        traces = candidate["native_reward_traces"]
        assert len(traces) == count
        native_lengths = []
        for index, trace in enumerate(traces):
            assert len(trace) == 1 and trace[0]
            assert set(trace[0]) <= {0, 1}
            assert max(trace[0]) == scores[index]
            native_lengths.append(len(trace[0]))
        values = [scores[index] for index in range(count)]
        rows.append({"task": candidate["task"], "per_scene_native_scores": values,
                     "successes": sum(values), "episodes": count,
                     "native_step_counts": native_lengths,
                     "runner_seconds_including_prompt_cache": candidate["seconds"],
                     "peak_torch_allocated_bytes": candidate["max_memory_allocated_bytes"]})
    return {"identity": folder.name, "complete": True, "scenes": list(range(count)),
            "all_initial_observations_paired": True,
            "distinct_initial_observations": count,
            "exact_saved_tensors": 967,
            "frozen_parameter_assertion_passed_before_final_receipt": True,
            "candidates": rows,
            "five_scene_engineering_gate": (all(sum(row["per_scene_native_scores"][:5]) >= 4
                                                 for row in rows) if count >= 5 else None),
            "outer_process_seconds": clock["ended"] - clock["started"],
            "interpretation": "Native motor qualification only; repeated scene indices are not independent held-out evidence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.folder)
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
