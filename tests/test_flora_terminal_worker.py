"""Persistent reward engine reuse, official numerical output and reader cleanup."""
import json
from types import SimpleNamespace as NS

import pytest

from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker


def backend(batch_size=1, gpu_memory_utilization=.78):
    state = NS(loads=0, processors=0, released=0, requests=[], fail=False, positions=[], batch_lengths=[])

    class Collage:
        num_frames, fps, duration_s = 751, 20., 37.55

        def __init__(self, videos):
            assert videos == ["/native/video.mp4"]

        def frames(self, indices):
            return ["selected-native-frames", indices]

        def release(self):
            state.released += 1

    class LLM:
        def get_tokenizer(self):
            return NS(decode=lambda tokens: str(tokens[0]))

        def generate(self, requests, params, *, use_tqdm):
            assert not use_tqdm and params.temperature == 0.
            state.requests.extend(requests)
            state.batch_lengths.append(len(requests))
            if state.fail:
                raise RuntimeError("inference failed")
            return [NS(outputs=[NS(text="3", token_ids=[3],
                logprobs=[{2: NS(logprob=-1.4), 3: NS(logprob=-.3)}])]) for _ in requests]

    def load_llm(path, args, **kwargs):
        assert path == "/pinned/model" and vars(args) == dict(tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=batch_size, seed=0)
        assert kwargs == dict(max_model_len=16384,
            mm_processor_kwargs=dict(do_sample_frames=False, size="official-size"), max_logprobs=20)
        state.loads += 1
        return LLM()

    def load_processor(path):
        state.processors += 1
        return "official-processor"

    def probabilities(positions, score):
        state.positions.append(positions)
        assert score == 3
        return {"1": 0., "2": .25, "3": .75, "4": 0., "5": 0.}, 2.75

    api = NS(core=NS(Collage=Collage, TERMINAL_TOP_LOGPROBS=20,
        terminal_frame_indices=lambda count, fps: [0, count - 1],
        terminal_max_tokens=lambda rationale: 16, terminal_regex=lambda rationale: "[1-5]",
        terminal_messages=lambda task, rationale: [task],
        video_metadata=lambda indices, fps, count: dict(indices=indices, fps=fps, count=count),
        parse_terminal=lambda text, rationale: (int(text), None), score_probabilities=probabilities),
        engine=NS(COLLAGE_VIDEO_SIZE="official-size", resolve_model=lambda *a, **kw: "/pinned/model",
            load_processor=load_processor, load_llm=load_llm,
            render=lambda processor, messages, thinking: messages[0],
            write_record=lambda stream, row: stream.write(json.dumps(row) + "\n")),
        SamplingParams=lambda **kwargs: NS(**kwargs), StructuredOutputsParams=lambda **kwargs: NS(**kwargs))
    return api, state


def test_engine_stays_loaded_across_batches_and_uses_official_grade(tmp_path):
    api, state = backend()
    worker = TerminalWorker("/pinned/model", _backend=api)
    for index in range(2):
        rows = worker.score_jobs([dict(id=str(index), task="open drawer", videos=["/native/video.mp4"])],
                                 tmp_path / f"batch-{index}.jsonl")
        assert rows[0]["expected_score"] == 2.75 and rows[0]["progress"] == .4375
        assert rows[0]["sampled_frames"] == 2
    assert state.loads == state.processors == worker.model_load_count == 1
    assert worker.completed_batches == state.released == 2
    assert state.requests[0] == state.requests[1]
    assert state.positions[0] == [("3", -.3, [("2", -1.4), ("3", -.3)])]


def test_failed_generation_releases_video_and_does_not_complete_batch(tmp_path):
    api, state = backend()
    worker = TerminalWorker("/pinned/model", _backend=api)
    state.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        worker.score_jobs([dict(id="1", task="open drawer", videos=["/native/video.mp4"])],
                          tmp_path / "partial.jsonl")
    assert state.released == 1 and worker.completed_batches == 0
    state.fail = False
    assert len(worker.score_jobs([dict(id="2", task="open drawer", videos=["/native/video.mp4"])],
                                 tmp_path / "complete.jsonl")) == 1
    assert state.loads == 1 and state.released == 2


def test_batched_generation_preserves_job_order_official_scores_and_tail(tmp_path):
    api, state = backend(batch_size=4)
    worker = TerminalWorker("/pinned/model", batch_size=4, _backend=api)
    jobs = [dict(id=str(index), task=f"task-{index}", videos=["/native/video.mp4"]) for index in range(10)]
    rows = worker.score_jobs(jobs, tmp_path / "batch.jsonl")
    assert state.batch_lengths == [4, 4, 2] and state.released == 10
    assert [row["id"] for row in rows] == [row["id"] for row in jobs]
    assert [request["prompt"] for request in state.requests] == [row["task"] for row in jobs]
    assert all(row["expected_score"] == 2.75 and row["progress"] == .4375 for row in rows)
    assert state.loads == worker.completed_batches == 1


def test_failed_batched_generation_releases_every_prepared_video(tmp_path):
    api, state = backend(batch_size=24)
    worker = TerminalWorker("/pinned/model", batch_size=24, _backend=api)
    state.fail = True
    jobs = [dict(id=str(index), task="open drawer", videos=["/native/video.mp4"]) for index in range(24)]
    with pytest.raises(RuntimeError, match="inference failed"):
        worker.score_jobs(jobs, tmp_path / "partial-24.jsonl")
    assert state.released == 24 and worker.completed_batches == 0


def test_explicit_engine_memory_budget_keeps_official_prompt_and_graded_output(tmp_path):
    api_default, default = backend(batch_size=4)
    api_lower, lower = backend(batch_size=4, gpu_memory_utilization=.65)
    a = TerminalWorker('/pinned/model', batch_size=4, _backend=api_default)
    b = TerminalWorker('/pinned/model', batch_size=4, gpu_memory_utilization=.65, _backend=api_lower)
    jobs = [dict(id=str(index), task='open drawer', videos=['/native/video.mp4']) for index in range(8)]
    first = a.score_jobs(jobs, tmp_path / 'default.jsonl')
    second = b.score_jobs(jobs, tmp_path / 'lower.jsonl')
    assert a.gpu_memory_utilization == .78 and b.gpu_memory_utilization == .65
    assert first == second and default.requests == lower.requests
    assert all(row['expected_score'] == 2.75 and row['progress'] == .4375 for row in second)


def test_prefix_scoring_never_samples_future_frames_or_reports_full_duration(tmp_path):
    api,state=backend(batch_size=2)
    worker=TerminalWorker('/pinned/model',batch_size=2,_backend=api)
    jobs=[dict(id=f'prefix-{step}',task='open drawer',videos=['/native/video.mp4'],
               prefix_controls=step,initial_frame_included=True) for step in (50,100)]
    rows=worker.score_jobs(jobs,tmp_path/'prefixes.jsonl')
    assert [row['observed_through'] for row in rows]==[50,100]
    assert [row['duration_s'] for row in rows]==[2.55,5.05]
    for step,request in zip((50,100),state.requests,strict=True):
        frames,metadata=request['multi_modal_data']['video']
        assert frames[1]==[0,step] and metadata['count']==step+1
    assert state.released==2


@pytest.mark.parametrize('changes',[dict(prefix_controls=751,initial_frame_included=True),
                                  dict(prefix_controls=50,initial_frame_included=False)])
def test_invalid_prefix_releases_reader_before_inference(tmp_path,changes):
    api,state=backend();worker=TerminalWorker('/pinned/model',_backend=api)
    with pytest.raises(ValueError):
        worker.score_jobs([dict(id='bad',task='open',videos=['/native/video.mp4'],**changes)],tmp_path/'bad.jsonl')
    assert state.released==1 and not state.requests
