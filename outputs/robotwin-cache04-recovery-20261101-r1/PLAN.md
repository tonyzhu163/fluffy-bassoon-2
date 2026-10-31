# Cache04 closed-case recovery correction

This is the same cache04 task and aggregate600GPU-s/2GiB lineage. No new GPU
budget, source/label/fit profile, allocator, framework or science. Source06 and
its84 actual passing test bodies remain immutable. Latestmain753bf542 was
reviewed; this fresh detached successor rebases the same code. The7eef publisher
on main is byte-identical SHA4850b43b6272ba6e8e1f13b4b017f8856f469a45ac2bc91042ff2d406c5107f8.
Its publish_cursor and checkpoint capture/restore APIs are reused unchanged.

The only resumable boundaries are BEFORE the whole04 case and AFTER the whole
04 case. There is no per-sample-prefix resume. Following actual model admission
and before any04 encode, publish the initial full cursor/RNG checkpoint, then
an immutable boundary-before-phone04.json copy. Completed payload files and
child directory entries are fsynced before the completed RNG/cursor is published.
The completed immutable boundary-after-phone04.json binds all13 actual source
chunks/400 historical native actions and the exact payload receipt. The root
partial-cache.json is emitted only after that completed boundary exists. The
full12 sealer now refuses raw cache04 payloads without this boundary proof.

Each snapshot binds exact code/profile/bank/runtime hashes, the cache root and
original clock contract, all paid11 receipt hashes, their completed cursor/full
checkpoint and actual successful allocation closeout. Paid11 payloads are not
loaded or encoded by the new worker. They remain readonly inputs for the later
normal full12 sealer. The same aggregate600 lineage includes every prior actual
charge/open reservation; a new child never resets it. Whole-root2GiB accounting,
source-ineligible status and all existing masks/labels/fit behavior remain.

The bundle has all Python/NumPy-global/TorchCPU/CUDA states. This model-only
cache consumes the captured Gaussian values directly and has no live steering
sampler. Its explicit private/motor generator is a canonical PCG64 replay cursor:
it is restored from the source first-before state, and at whole-case completion
is moved to the bound final-after state. It never draws replacement noise.
Private is explicitly an alias of motor_rng; the role string and both canonical
endpoint states are retained with the full SHA-bound13-chunk state chain. The
cache's actual model randomness is covered by the global four-domain snapshot.
This is truthful replay-state preservation, not a claim that caching generated
new controller actions.

On interruption, keep the published initial pointer/RNG and every partial file
and orphan generation. A new child must bind cache04_resume.cursor, closeout and
diagnosis refs. The old allocation must be truly finished in the same original
clock; the diagnosis must name the cursor SHA. Code/profile/bank/paid11/runtime
identity must match exactly. After the fresh model is initialized and qualified,
restore all saved global/private/motor state, publish a new initial boundary,
and re-encode only04. A stale before-boundary copy cannot resume if a newer
completed cursor has been published.

If completion succeeded but root receipt publication failed, reconcile the
complete boundary on allocated CPU using DATA_OPERATION=reconcile-cache04 and
an input request containing cache04_cursor:{path,sha256}. This loads/hashes the
owned recovery and payloads and emits a fresh receipt; no GPU/reporting retry.
An interrupted completed-pointer publication preserves the initial pointer and
orphan complete snapshot; it does not claim a committed sample prefix. Existing
backend/owner teardown and actual scheduler-cost reporting remain authoritative.

New request fields for cache04:

- partial11_cursor:{path,sha256}, with complete_partial11/all11/clean teardown;
- partial11_closeout:{path,sha256}, binding successful actual source/attempt/cost;
- cache_receipts:[{path,sha256},...], exactly the paid77 records;
- cache04_resume:null initially, otherwise {cursor,closeout,diagnosis} SHA refs.

The profileefcf,84 passing bodies, numerical five curve functions, source bank
semantics and canonical readout-v24fcf are unchanged. Only the new focused
recovery suite should run under the owner's explicit LINKED CPU-only grant;
old1775/1800 accounting stays as recorded. No model GPU work until the recovery
contract is actually qualified. Existing owner alone deploys/admit/launches and
monitors resources; source author runs only Mac static checks.

```sh
python -m pytest -q tests/test_robotwin_cache04_recovery.py --junitxml="$RECEIPTS/cache04-recovery.junit.xml"
```

Use the same diagnostic GPU-PAYLOAD cache04 command with the new source pin,
updated runtime hashes and actual request fields. The payload still books and
closes no allocation or clock. Actual data/parity/model capacity/cost remain
required, and no new success/source/student eligibility is implied.
