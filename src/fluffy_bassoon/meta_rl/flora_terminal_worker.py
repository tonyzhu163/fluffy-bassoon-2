"""Persistent official Flora terminal engine, shared by calibration and PPO.

The pinned rewarder owns prompts, frame selection, probabilities and parsing.
This adapter only retains its engine between complete batches. Imports stay
lazy so the policy process never imports vLLM or the scorer's dependencies.
"""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def _official():
    from rewarder.batch_utils import core, engine
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    return SimpleNamespace(core=core, engine=engine, SamplingParams=SamplingParams,
                           StructuredOutputsParams=StructuredOutputsParams)


class TerminalWorker:
    """Load once; use the exact terminal_reward CLI recipe for every video."""
    def __init__(self, model, *, _backend=None):
        api = _official() if _backend is None else _backend
        self.api = api
        args = Namespace(tensor_parallel_size=1, gpu_memory_utilization=.78,
                         max_num_seqs=1, seed=0)
        path = api.engine.resolve_model(str(model), None, weights=True)
        self.processor = api.engine.load_processor(path)
        self.llm = api.engine.load_llm(path, args, max_model_len=16384,
            mm_processor_kwargs={"do_sample_frames": False, "size": api.engine.COLLAGE_VIDEO_SIZE},
            max_logprobs=api.core.TERMINAL_TOP_LOGPROBS)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = api.SamplingParams(temperature=0.,
            max_tokens=api.core.terminal_max_tokens(False),
            logprobs=api.core.TERMINAL_TOP_LOGPROBS,
            structured_outputs=api.StructuredOutputsParams(regex=api.core.terminal_regex(False)))
        self.model_load_count = 1
        self.completed_batches = 0

    def score_jobs(self, jobs, output):
        """Append no partial labels to PPO; retain a partial JSONL on failure."""
        core, engine = self.api.core, self.api.engine
        rows = []
        with Path(output).open("x") as stream:
            # Official max_num_seqs=1: preparing one collage at a time preserves
            # requests and avoids holding every rollout's video reader open.
            for job in jobs:
                collage = core.Collage(job["videos"])
                try:
                    indices = core.terminal_frame_indices(collage.num_frames, collage.fps)
                    prompt = engine.render(self.processor,
                        core.terminal_messages(job["task"], False), thinking=False)
                    request = dict(prompt=prompt, multi_modal_data={"video": (
                        collage.frames(indices),
                        core.video_metadata(indices, collage.fps, collage.num_frames))},
                        mm_processor_kwargs={"do_sample_frames": False})
                    outputs = self.llm.generate([request], self.params, use_tqdm=False)
                    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                        raise ValueError("Exactly one official terminal completion per video required")
                    completion = outputs[0].outputs[0]
                    score, rationale = core.parse_terminal(completion.text, False)
                    positions = [(self.tokenizer.decode([token]), float(candidates[token].logprob),
                        [(self.tokenizer.decode([i]), float(candidate.logprob))
                         for i, candidate in candidates.items()])
                        for token, candidates in zip(completion.token_ids, completion.logprobs)]
                    probabilities, expected = core.score_probabilities(positions, score)
                    row = dict(id=job["id"], task=job["task"], score=score,
                        expected_score=round(expected, 4), progress=round((expected - 1.) / 4., 4),
                        probabilities={key: round(value, 4) for key, value in probabilities.items()},
                        rationale=rationale, duration_s=round(collage.duration_s, 3),
                        sampled_frames=len(indices), output=completion.text)
                    engine.write_record(stream, row)
                    rows.append(row)
                finally:
                    collage.release()
        self.completed_batches += 1
        return rows
