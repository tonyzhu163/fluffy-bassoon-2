"""ONE flat metadata pass over162 explicit rawRPC directories. No payload/hash read."""
from pathlib import Path
import json,os,re,stat,time
root=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1/data')
started=time.monotonic();deadline=started+180;rows=[];unknown=[];total=0;file_stats=0;error=None;complete=False
pattern=re.compile(r'[0-9]+-(reset|inference|outcome)\.pkl\.gz')
try:
 for ordinal in range(162):
  if time.monotonic()>=deadline:raise TimeoutError('Original180second metadata deadline')
  path=root/f'episode-{ordinal:03d}'/'raw-rpc';mode=os.lstat(path).st_mode
  if not stat.S_ISDIR(mode):unknown.append(dict(path=str(path),reason='known rawRPC path is not a regular directory'));raise ValueError('Unknown directory class')
  entries=[];counts={'reset':0,'inference':0,'outcome':0};bytes_this=0
  with os.scandir(path) as stream:
   for entry in stream:
    if time.monotonic()>=deadline:raise TimeoutError('Original180second metadata deadline')
    file_stats+=1
    if file_stats>5346:raise ValueError('File count exceeds source/counter-proved5346')
    info=entry.stat(follow_symlinks=False);match=pattern.fullmatch(entry.name)
    if not stat.S_ISREG(info.st_mode) or match is None or info.st_size<1:
     unknown.append(dict(path=str(path/entry.name),mode=info.st_mode,bytes=info.st_size,reason='nonfile/link/unknownname/emptyrecord'));raise ValueError('Unknown rawRPC file class')
    kind=match.group(1);counts[kind]+=1;bytes_this+=info.st_size
    entries.append(dict(name=entry.name,kind=kind,bytes=info.st_size,mtime_ns=info.st_mtime_ns))
  if counts['reset']!=1 or counts['inference']!=counts['outcome'] or (ordinal<161 and counts['inference']<1) or (ordinal==161 and counts['inference']!=0):
   unknown.append(dict(path=str(path),counts=counts,reason='raw record count differs from closed/failed protocol'));raise ValueError('Unexpected per-role rawRPC structure')
  total+=bytes_this;rows.append(dict(ordinal=ordinal,path=str(path),counts=counts,bytes=bytes_this,entries=sorted(entries,key=lambda x:x['name'])))
 counts={k:sum(row['counts'][k] for row in rows) for k in ['reset','inference','outcome']}
 if counts['reset']!=162 or counts['inference']>2592 or file_stats!=2*counts['inference']+162:raise ValueError('Cumulative rawRPC count/counter proof differs')
 complete=True
except BaseException as e:error=f'{type(e).__name__}: {e}'
print(json.dumps(dict(format='assessment_flat_rawRPC_size_metadata_v1',status='complete_exact_rawRPC_bytes' if complete and not unknown else 'held_partial_or_unknown',complete=complete,root=str(root),directory_scope='ONLY episode000..161/raw-rpc immediateentries, no recursion, no other dirs',completed_directories=len(rows),expected_directories=162,files_lstat=file_stats,maximum_files=5346,total_bytes_of_complete_directories=total,exact_total_rawRPC_bytes=total if complete else None,unfamiliar=unknown,error=error,rows=rows,elapsed_seconds=time.monotonic()-started,deadline_seconds=180,payload_bytes_read=0,payload_hashes_computed=0,journals_read=0,previous626files_restatted=0,new_allocation=False,models_or_arrays_loaded=False,raw1MiB_assumption_explicitly_rejected=True),indent=2))
