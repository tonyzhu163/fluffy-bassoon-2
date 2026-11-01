#!/usr/bin/env python
"""Measure exact official three-view processor tokens before engine construction.

This performs video decoding and CPU preprocessing only inside the admitted
Linux benchmark allocation. It never constructs a model or changes a prompt.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import time

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import digest, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    output = Path(profile['output'])
    started = time.monotonic()
    result = dict(completed=False, model_constructed=False, new_robot_actions=0,
                  source_commit=profile['source_commit'], profile_sha256=digest(profile),
                  requests=[])
    try:
        if (platform.system() != 'Linux' or not socket.gethostname().startswith('job-')
                or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
            raise RuntimeError('CPU child of the admitted Linux GPU benchmark required')
        from rewarder.batch_utils import core, engine
        processor = engine.load_processor(profile['model'])
        result.update(processor_class=type(processor).__name__,
                      video_processor_class=type(processor.video_processor).__name__,
                      official_visual_token_cap=core.MAX_VISUAL_TOKENS,
                      official_video_size=engine.COLLAGE_VIDEO_SIZE,
                      context_limit=profile['max_model_len'])
        for job in profile['jobs']:
            collage = core.Collage(job['videos'])
            try:
                if len(collage.videos) != 3:
                    raise ValueError('Exactly three synchronized native camera streams required')
                indices = core.terminal_frame_indices(collage.num_frames, collage.fps)
                prompt = engine.render(processor, core.terminal_messages(job['task'], False),
                                       thinking=False)
                encoded = processor(text=[prompt], videos=[collage.frames(indices)],
                    video_metadata=[core.video_metadata(indices, collage.fps, collage.num_frames)],
                    do_sample_frames=False, size=engine.COLLAGE_VIDEO_SIZE, return_tensors='pt')
                grids = encoded['video_grid_thw'].tolist()
                total = int(encoded['input_ids'].shape[-1])
                visual = int((encoded['input_ids'] == processor.video_token_id).sum().item())
                expected = sum(t*h*w for t,h,w in grids) // processor.video_processor.merge_size**2
                row = dict(id=job['id'], native_instruction=job['task'], sampled_frames=len(indices),
                    native_fps=collage.fps, native_frame_count=collage.num_frames,
                    source_dimensions=[[v.width, v.height] for v in collage.videos],
                    rectangles=collage.rectangles, collage_dimensions=[collage.width, collage.height],
                    video_grid_thw=grids, visual_tokens=visual, total_input_tokens=total,
                    nonvisual_tokens=total-visual, completion_reserve=core.terminal_max_tokens(False))
                result['requests'].append(row)
                if (len(indices) != 64 or visual != expected or visual > core.MAX_VISUAL_TOKENS
                        or total + row['completion_reserve'] > profile['max_model_len']):
                    raise ValueError('Official full-frame request does not fit the declared context; no truncation or fallback')
                del encoded
            finally:
                collage.release()
        result['completed'] = True
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write_new(output, result)


if __name__ == '__main__':
    main()
