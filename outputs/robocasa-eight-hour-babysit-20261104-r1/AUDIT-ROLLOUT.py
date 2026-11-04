"""Independent CPU audit of completed native six-episode artifacts."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time


def main():
    assert sys.platform == 'linux'
    import cv2
    import numpy as np
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    began = time.time()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def checked(ref):
        path = Path(ref['path'])
        assert digest(path) == ref['sha256'], path
        return path
    result = json.loads(args.result.read_text())
    episodes, chunks, controls, frames, videos = 0, 0, 0, 0, []
    policies, bindings, case_attempts = set(), set(), {}
    for ref in result['references']:
        saved = torch.load(checked(ref), map_location='cpu', weights_only=False)
        case, attempt, horizon = saved['case'], saved['attempt'], saved['horizon']
        assert saved['completed'] and len(saved['chunks']) * 50 == horizon
        key = json.dumps(case, sort_keys=True)
        case_attempts.setdefault(key, []).append(attempt)
        assert saved['memory_entry']['length'] == (attempt - 1) * (horizon // 50)
        assert saved['memory_exit']['length'] == attempt * (horizon // 50)
        assert saved['memory_exit']['previous_done'] and not saved['memory_exit']['pending']
        policies.add(saved['policy_weights_sha256'])
        bindings.add(json.dumps(saved['binding'], sort_keys=True))
        initial = torch.load(checked(saved['initial']), map_location='cpu', weights_only=False)
        pair = initial['evidence']['pairing']
        assert pair['matched'] and pair['physical_state_exact'] and pair['public_state_exact']
        assert set(pair['images']) == {'main_images', 'wrist_images', 'extra_view_images'}
        assert all(row['max_absolute_delta'] <= 2 for row in pair['images'].values())
        truth = json.loads(checked(saved['truth']).read_text())
        assert truth['case'] == case
        assert truth['simulator_success_once'] == saved['success_once']
        assert truth['simulator_final_success'] == saved['success_at_end']
        assert truth['first_success_step'] == saved['first_success_step']
        assert truth['native_counts']['controls'] - truth['native_counts']['reset_controls'] == horizon
        first_success, previous_exit = None, saved['memory_entry']
        for index, cref in enumerate(saved['chunks']):
            with gzip.open(checked(cref), 'rb') as stream:
                row = pickle.load(stream)
            assert row['status'] == 'completed' and row['binding'] == saved['binding']
            assert row['case'] == case and row['attempt'] == attempt and row['chunk'] == index
            identity = dict(version='frozen_steering_actual_attempt_noise_v1', case=case,
                            attempt=attempt, chunk=index)
            seed = int.from_bytes(hashlib.sha256(json.dumps(identity, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).digest()[:8], 'little')
            noise = np.random.Generator(np.random.PCG64(seed)).standard_normal((50,32)).astype(np.float32)
            assert np.array_equal(noise, row['native_noise'])
            assert row['native_noise_provenance'] == dict(identity, seed=seed,
                law='PCG64 standard_normal float64 cast float32')
            physical = np.asarray(row['requested_commands'], np.float32).copy()
            physical[:,[6,11]] = np.where(physical[:,[6,11]] > .5, 1., -1.)
            np.testing.assert_array_equal(physical, row['executed_commands'])
            np.testing.assert_array_equal(physical, row['dispatched_commands'])
            mean = np.asarray(saved['interface']['mean'], np.float32)
            std = np.asarray(saved['interface']['std'], np.float32)
            np.testing.assert_array_equal(((physical-mean)/std).reshape(600), row['executed_control'])
            before, after = row['native_counts_before'], row['native_counts_after']
            assert after['controls'] - before['controls'] == 50
            assert after['reset_controls'] == before['reset_controls']
            assert row['done'] == (index == horizon // 50 - 1)
            assert row['memory_entry'] == (saved['memory_entry'] if index == 0 else None)
            newly_seen = first_success is None and row['first_success_step'] is not None
            if row['first_success_step'] is not None:
                first_success = row['first_success_step'] if first_success is None else first_success
                assert first_success == row['first_success_step']
            assert row['reward'] == float(newly_seen)
            chunks += 1
        assert first_success == saved['first_success_step']
        video = json.loads(checked(saved['video']).read_text())
        assert video['completed'] and video['horizon'] == horizon
        assert video['frame_clock']['frame_count'] == horizon + 1
        for vref in video['videos']:
            path = checked(vref)
            capture = cv2.VideoCapture(str(path)); count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                assert frame.shape == (256,256,3)
                count += 1
            capture.release()
            assert count == horizon + 1, (path,count,horizon)
            frames += count
            videos.append(dict(path=str(path), frames=count, sha256=vref['sha256']))
        episodes += 1
        controls += horizon
    assert all(sorted(values) == list(range(1,7)) for values in case_attempts.values())
    assert len(policies) == len(bindings) == 1
    assert result['optimizer_updates'] == 0 and result['native_parameters_frozen']
    assert result['controls_charged'] >= controls
    audit = dict(passed=True, checked_shanghai=datetime.datetime.now(datetime.timezone(
        datetime.timedelta(hours=8))).isoformat(), result_sha256=digest(args.result),
        episodes=episodes, physical_controls=controls, complete_chunks=chunks,
        decoded_videos=len(videos), video_frames=frames, all_commands_and_draws_reproduced=True,
        all_initial_physical_public_states_match=True, maximum_allowed_RGB_component_delta=2,
        SGD=0, all_prior_history_retained=True, elapsed_CPU_seconds=time.time()-began, videos=videos)
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'videos'}))


if __name__ == '__main__':
    main()
