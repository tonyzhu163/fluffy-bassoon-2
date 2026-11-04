"""Compare archived native rollout RNG and actual initial-noise tensors on Linux CPU."""
import argparse
import datetime
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import tarfile

import torch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor_sha(tensor):
    return sha(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', default='PickPlaceDrawerToCounter')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.root / f'CORPUS-ARCHIVE-{args.task}-R1-RECLAIMED.json'
    receipt_data = receipt_path.read_bytes()
    receipt = json.loads(receipt_data)
    manifest_data = Path(receipt['manifest']).read_bytes()
    assert sha(manifest_data) == receipt['manifest_sha256']
    manifest = json.loads(manifest_data)
    offsets, offset = {}, 0
    # The archive writer uses sorted regular PAX members, uid/gid zero and
    # mtime/mode/size recorded in this manifest. Reconstruct offsets without
    # streaming 100+ GB. Every selected payload must pass its recorded SHA256.
    for name, item in sorted(manifest['files'].items()):
        member = tarfile.TarInfo(name)
        member.size, member.mode = item['bytes'], item['mode']
        member.mtime = item['mtime_ns'] / 1e9
        header_size = len(member.tobuf(format=tarfile.PAX_FORMAT))
        offsets[name] = offset + header_size
        offset += header_size + ((item['bytes'] + 511) // 512) * 512
    rows, source_files, noise_cache = [], [], {}
    with Path(receipt['archive']).open('rb') as archive:
        def load(phase, rank, ordinal):
            name = f'{args.task}/{phase}/sampling/rank-{rank:02d}/{ordinal:05d}.pt.gz'
            item = manifest['files'][name]
            archive.seek(offsets[name])
            payload = archive.read(item['bytes'])
            assert len(payload) == item['bytes'] and sha(payload) == item['sha256'], name
            record = torch.load(io.BytesIO(gzip.decompress(payload)), map_location='cpu', weights_only=False)
            assert record['rank'] == rank and record['prediction_ordinal'] == ordinal
            source_files.append(dict(member=name, offset=offsets[name], bytes=len(payload), sha256=sha(payload)))
            return record

        def describe(record, phase, label):
            before = record['rng_before']
            fields = record['forward_sampling_fields']
            chain = fields.get('chains')
            cuda = before['cuda']
            row = dict(phase=phase, label=label, rank=record['rank'],
                ordinal=record['prediction_ordinal'], mode=record['mode'],
                collection_iteration=record['collection_iteration'],
                sampling_record=record['sampling_record'],
                cuda_states=[dict(device_index=i, bytes=t.numel(), sha256=tensor_sha(t),
                    seed_le64=int.from_bytes(t.numpy().tobytes()[:8], 'little'))
                    for i,t in enumerate(cuda)],
                torch_cpu_rng_sha256=tensor_sha(before['torch']),
                python_rng_sha256=sha(json.dumps(before['python']).encode()),
                denoising_indices=fields.get('denoise_inds', torch.empty(0)).tolist())
            if chain is not None:
                initial = chain[:, 0]
                row.update(initial_noise_shape=list(initial.shape), initial_noise_sha256=tensor_sha(initial),
                    initial_noise_lane_sha256=[tensor_sha(lane) for lane in initial])
                gaussian = [g for g in record['actual_gaussian_parameters'] if g['noise_method']=='flow_sde']
                assert len(gaussian)==1
                gaussian = gaussian[0]
                index = int(gaussian['index'][0])
                assert bool((gaussian['index']==index).all()) and bool((gaussian['std']>0).all())
                innovation = (chain[:,index+1].double()-gaussian['mean'].double())/gaussian['std'].double()
                row.update(stochastic_denoising_index=index, sde_innovation_sha256=tensor_sha(innovation),
                    innovation_note='Recovered from rounded float32 next-state/mean/std, not bitwise original Gaussian draw')
                noise_cache[(phase,label,record['rank'])] = (initial,innovation,index)
            rows.append(row)

        for phase in ['q1-r1', 'production-r1']:
            for rank in range(8):
                first = load(phase, rank, 0)
                describe(first, phase, 'ordinal_zero')
                ordinal = 0
                record = first
                if record['mode'] != 'train' or record['terminal_value_only']:
                    # Native before-panel has 12 H50 decisions at H600.
                    ordinal = 12
                    record = load(phase, rank, ordinal)
                while record['mode'] != 'train' or record['terminal_value_only']:
                    ordinal += 1
                    assert ordinal < 40
                    record = load(phase, rank, ordinal)
                describe(record, phase, 'first_training')
                describe(load(phase, rank, ordinal+1), phase, 'next_training')

    summaries = []
    for phase, label in itertools.product(['q1-r1','production-r1'],
            ['ordinal_zero','first_training','next_training']):
        group = sorted((r for r in rows if r['phase']==phase and r['label']==label), key=lambda r:r['rank'])
        assert len(group)==8
        initial = [r.get('initial_noise_sha256') for r in group]
        correlations = []
        for ra,rb in itertools.combinations(group,2):
            ka,kb=(phase,label,ra['rank']),(phase,label,rb['rank'])
            if ka not in noise_cache or kb not in noise_cache:
                continue
            ia,ea,ta = noise_cache[ka]
            ib,eb,tb = noise_cache[kb]
            def correlation(a,b):
                a,b=a.double().flatten(),b.double().flatten()
                a,b=a-a.mean(),b-b.mean()
                return float(torch.dot(a,b)/(a.norm()*b.norm()))
            correlations.append(dict(ranks=[ra['rank'],rb['rank']], initial_noise_correlation=correlation(ia,ib),
                selected_sde_innovation_correlation=correlation(ea,eb), same_denoising_index=ta==tb))
        summaries.append(dict(phase=phase, label=label, ordinals=[r['ordinal'] for r in group],
            cuda_state_list_unique=len({tuple(t['sha256'] for t in r['cuda_states']) for r in group}),
            cuda_seed_sets=[list({t['seed_le64'] for t in r['cuda_states']}) for r in group],
            all_cuda_states_nonempty=all(r['cuda_states'] for r in group),
            initial_noise_unique=len(set(initial)) if all(initial) else None,
            python_rng_unique=len({r['python_rng_sha256'] for r in group}),
            denoising_index_per_rank=[r['denoising_indices'][0][0] if r['denoising_indices'] else None for r in group],
            noise_pair_correlations=correlations))
    report=dict(completed=True, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task=args.task, runtime='Linux CPU; no model calls, new sampling or SGD',
        script_sha256=sha(Path(__file__).read_bytes()), receipt=str(receipt_path),
        receipt_sha256=sha(receipt_data), archive=receipt['archive'],
        archive_sha256_from_verified_closeout=receipt['archive_sha256'],
        manifest=receipt['manifest'], manifest_sha256=sha(manifest_data),
        selected_member_hashes_verified=True, source_files=source_files, summaries=summaries, rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    main()
