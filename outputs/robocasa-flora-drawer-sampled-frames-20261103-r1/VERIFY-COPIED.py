"""Verify downloaded artifact bytes and reconstruction receipts; no model execution."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
archive = Path('/Users/mowenjie/Downloads/drawer-sampled-frames-r1.zip')
source = '382ec480955e61844cba38bc512a8e6ae14f15bf'
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_text())
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('MANIFEST.json'))
    assert manifest['source_commit'] == source and len(manifest['files']) == 33
    for name,digest in manifest['files'].items():
        assert Path(name).name == name
        data = z.read(name)
        assert sha(data) == digest,name
        path = root/name
        if path.exists(): assert path.read_bytes() == data,name
        else: path.write_bytes(data)
    (root/'MANIFEST.json').write_bytes(z.read('MANIFEST.json'))
receipt = read(root/'RECEIPT.json')
assert receipt['completed'] and receipt['source_commit'] == source
assert receipt['case_count'] == len(receipt['cases']) == 4
assert receipt['decoded_collages'] == 256 and receipt['pages'] == 32
assert all(receipt[k] == 0 for k in ('model_calls','simulator_calls','policy_controls','motor_calls','optimizer_steps'))
expected = {'000112':350,'000256':750,'000376':750,'000128':450}
assert {c['case']:c['prefix_controls'] for c in receipt['cases']} == expected
for case in receipt['cases']:
    frames,pages = case['sampled_frames'],case['pages']
    assert len(frames) == 64 and len(pages) == 8 and case['page_pixel_roundtrip']
    assert case['raw_frame_shape'] == [320,960,3]
    assert [f['ordinal'] for f in frames] == list(range(64))
    assert [f['control'] for f in frames] == [round(i*case['prefix_controls']/63) for i in range(64)]
    assert [i for page in pages for i in page['samples']] == list(range(64))
    for page in pages:
        assert sha((root/page['file']).read_bytes()) == page['sha256']
    assert all(len(f['rgb_sha256']) == 64 for f in frames)
report = dict(completed=True,source_commit=source,archive_sha256=sha(archive.read_bytes()),
    verified_files=33,cases=4,collages=256,pages=32,cpu_seconds=receipt['seconds'],
    expected_sampler_indices_match=True,model_calls=0,simulator_calls=0,
    local_check='Downloaded artifact bytes and JSON only')
(root/'COPIED-HASHES.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
