"""Qualify bounded rubric prompts and camera arrays; never load model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = 'robocasa-flora-rubric-interface-20261103-r1'
PREVIOUS = ROOT / 'outputs/robocasa-flora-current-state-20261103-r2'
SELECT = {('000112', 350), ('000128', 450), ('000216', 300), ('000376', 750)}
JOBS_SHA = 'e06662b1021e659aad10e7070326121cf6460dd3eb2dcf9f4ad522ce0bc276ce'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main(code):
    started = time.time()
    assert platform.system() == 'Linux'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('PYTHONNOUSERSITE') == '1'
    pin = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    assert pin == os.environ['TASK_CODE_PIN']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
    assert sha(PREVIOUS / 'PUBLIC-JOBS.json') == JOBS_SHA
    public = json.loads((PREVIOUS / 'PUBLIC-JOBS.json').read_text())
    selected = [j for j in public if j['arm'] == 'prefix_grade'
                and (j['id'].split('single-case-')[1][:6], j['control']) in SELECT]
    assert len(selected) == len(SELECT)
    inputs = json.loads((code / 'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json').read_text())
    assert subprocess.check_output(['git', '-C', inputs['official_code'], 'rev-parse', 'HEAD'], text=True).strip() == inputs['official_code_commit']
    previous_inputs = json.loads((code / 'outputs/robocasa-flora-current-state-20261103-r2/INPUTS.json').read_text())
    for path, expected in previous_inputs['scorer_sources'].items():
        assert sha(path) == expected
    receipt = inputs['model_receipt']
    assert sha(receipt['path']) == receipt['sha256']
    assert json.loads(Path(receipt['path']).read_text())['completed']
    space = os.statvfs(ROOT)
    assert space.f_bavail * space.f_frsize >= 8 << 30
    out = ROOT / 'outputs' / RUN
    out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from transformers.video_utils import VideoMetadata
    from rewarder.batch_utils import core, engine
    torch.set_num_threads(4)
    rubric_source = next((Path(inputs['official_code']) / 'rewarder').rglob('*rubric*.py'))
    processor = engine.load_processor(inputs['model'])
    print('PROCESSOR_READY', flush=True)
    records = []
    for job in selected:
        side = job['task'].split()[2]
        assert side in ('left', 'right')
        target = (f'the very top drawer directly below the countertop on the {side} '
                  'of the robot at reset, in the same initially selected drawer stack')
        steps = [
            dict(action=f'At the LAST visible time, identify and track {target}. '
                 'A lower drawer is not this target. Award 0.5 only if target identity '
                 'is visually established; otherwise award 0.',
                 grading=[dict(label='not established', points=0), dict(label='established', points=.5)]),
            dict(action='At the LAST visible time, the identified target drawer must '
                 'have outward displacement at least 0.5225 times cabinet-fixture depth. '
                 'This is not 95% of mechanical travel. Holding the handle is allowed. '
                 'Do not count a different drawer, earlier opening, or slight movement. '
                 'Award 0.5 only when this current target opening is visibly established; '
                 'otherwise award 0, including uncertain or unidentified target.',
                 grading=[dict(label='not established', points=0), dict(label='established', points=.5)]),
        ]
        for view in job['videos']:
            assert sha(view['path']) == view['sha256']
            info = core.probe_video(view['path'])
            assert info.num_frames == 751 and info.fps == 20.
        for mode in ('collage64', 'cameras32'):
            if mode == 'collage64':
                indices = list(core.terminal_frame_indices(job['control'] + 1, 20.))
                reader = core.Collage([v['path'] for v in job['videos']])
                try:
                    videos = [reader.frames(indices)]
                finally:
                    reader.release()
                cameras = ['a synchronized left scene, wrist, and right scene camera collage, in that left-to-right order']
            else:
                indices = np.rint(np.linspace(0, job['control'], 32)).astype(int).tolist()
                videos = []
                for view in job['videos']:
                    decoded = core.read_frames(view['path'], indices)
                    videos.append(np.stack([decoded[i] for i in indices]))
                cameras = ['the left scene camera', 'the wrist camera', 'the right scene camera']
            assert indices[0] == 0 and indices[-1] == job['control']
            assert max(indices) <= job['control']
            metadata = core.video_metadata(indices, 20., job['control'] + 1)
            prompt = core.rubric_prompt(job['task'], steps, core.rubric_header('single-arm robot', cameras))
            messages = core.rubric_messages(prompt, [core.VIDEO_PLACEHOLDER] * len(videos))
            rendered = engine.render(processor, messages, thinking=True, vision_ids=True)
            meta = VideoMetadata(**{k: v for k, v in metadata.items() if k != 'do_sample_frames'})
            result = processor(text=[rendered], videos=[videos],
                video_metadata=[[meta for _ in videos]], do_sample_frames=False,
                size=engine.COLLAGE_VIDEO_SIZE, return_tensors='pt')
            tokens = result['input_ids'].shape[-1]
            assert tokens + 2048 <= 16384, (job['id'], mode, tokens)
            features = result['pixel_values_videos'].to(torch.bfloat16).contiguous()
            assert len(result['video_grid_thw']) == len(videos)
            record = dict(source_id=job['source_id'], mode=mode, indices=indices,
                cameras=cameras, camera_shapes=[list(v.shape) for v in videos],
                rgb_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in videos],
                metadata=metadata, messages=messages, prompt_tokens=tokens,
                output_token_reserve=2048, maximum_model_length=16384,
                rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                bf16_sha256=hashlib.sha256(features.view(torch.uint8).numpy().tobytes()).hexdigest(),
                grid=result['video_grid_thw'].tolist())
            records.append(record)
            print(json.dumps({k: record[k] for k in ('source_id', 'mode', 'prompt_tokens', 'grid')}), flush=True)
    assert core.parse_rubric('ANSWER: [0.5, 0.5]', steps) == ([.5, .5], True)
    assert core.split_thinking('<think>diagnostic</think>ANSWER: [0, 0]') == ('ANSWER: [0, 0]', 'diagnostic')
    assert not core.parse_rubric('no scores', steps)[1]
    assert not torch.cuda.is_initialized()
    write(out / 'PREPARED.json', records)
    write(out / 'CPU-CHECKS.json', dict(completed=True, source_commit=pin, records=len(records),
        prepared_sha256=sha(out / 'PREPARED.json'), seconds=time.time()-started,
        native_rubric_source=sha(rubric_source), official_code_commit=inputs['official_code_commit'],
        processor_only=True, model_weight_loads=0, cuda_initialized=False,
        gpu_seconds=0, policy_controls=0, optimizer_steps=0,
        actual_cli_video_reader_used=False, interpretation='Official prompt/template/parser; custom frozen causal arrays.'))
    print('RUBRIC_INPUT_QUALIFICATION_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    main(parser.parse_args().code)
