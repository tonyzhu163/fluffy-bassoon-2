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
    def __init__(self, model, *, batch_size=1, _backend=None):
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("Finite official inference batch size 1..32 required")
        self.batch_size = batch_size
        api = _official() if _backend is None else _backend
        self.api = api
        args = Namespace(tensor_parallel_size=1, gpu_memory_utilization=.78,
                         max_num_seqs=batch_size, seed=0)
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
            for start in range(0, len(jobs), self.batch_size):
                batch, prepared = jobs[start:start + self.batch_size], []
                try:
                    for job in batch:
                        collage = core.Collage(job["videos"])
                        prepared.append((job, collage, None))
                        indices = core.terminal_frame_indices(collage.num_frames, collage.fps)
                        prepared[-1] = (job, collage, indices)
                    requests = [dict(prompt=engine.render(self.processor,
                        core.terminal_messages(job["task"], False), thinking=False),
                        multi_modal_data={"video": (collage.frames(indices),
                            core.video_metadata(indices, collage.fps, collage.num_frames))},
                        mm_processor_kwargs={"do_sample_frames": False}) for job,collage,indices in prepared]
                    outputs = self.llm.generate(requests, self.params, use_tqdm=False)
                    if len(outputs) != len(batch) or any(len(output.outputs)!=1 for output in outputs):
                        raise ValueError("Exactly one official terminal completion per video required")
                    for (job,collage,indices), output in zip(prepared,outputs):
                        completion = output.outputs[0]
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
                    for _,collage,_ in prepared:collage.release()
        self.completed_batches += 1
        return rows
