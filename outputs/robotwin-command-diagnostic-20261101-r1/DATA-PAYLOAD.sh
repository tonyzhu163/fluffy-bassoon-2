#!/usr/bin/env bash
# Run only inside the owner's actual finite allocated Linux CPU/data clock.
set -euo pipefail
: "${DATA_OPERATION:?assemble or seal}"
: "${DATA_REQUEST:?actual absolute request file}"
: "${DATA_REQUEST_SHA:?exact SHA256}"
: "${DATA_OUTPUT:?fresh absolute output JSON}"
: "${OWNER_DATA_DEADLINE:?original absolute deadline; no post-load reset}"
python - <<'PY'
import os,sys
from pathlib import Path
if sys.platform!='linux':raise RuntimeError('Allocated Linux only')
from fluffy_bassoon.meta_rl.robotwin_replay_diagnosis import Audit
from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_bank as bank
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
audit=Audit(float(os.environ['OWNER_DATA_DEADLINE']))
r=audit.json(dict(path=os.environ['DATA_REQUEST'],sha256=os.environ['DATA_REQUEST_SHA']))
if os.environ['DATA_OPERATION']=='assemble':
    value=bank.assemble_bank(r['demos'],r['interface'],r['result_refs'],audit=audit)
elif os.environ['DATA_OPERATION']=='seal':
    value=bank.seal_full_cache(audit.json(r['bank']),r['cache_receipts'],deadline=audit.deadline)
else:raise ValueError('Only assemble or seal')
_atomic_json(Path(os.environ['DATA_OUTPUT']),value)
PY
