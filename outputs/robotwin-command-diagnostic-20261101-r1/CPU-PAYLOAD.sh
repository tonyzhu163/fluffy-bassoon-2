#!/usr/bin/env bash
# Component of the EXISTING owner's attested allocated Linux CPU clock.
# Does not allocate, reset a clock, launch GPU work or open a campaign ledger.
set -euo pipefail
: "${RECEIPTS:?fresh owned receipt directory}"
python -m pytest -q tests/test_robotwin_command_diagnostic.py --junitxml="$RECEIPTS/command-diagnostic.junit.xml"
# Include tests/test_robotwin_reactive_curve.py in the SAME pytest invocation
# only if its42 prepared cases still lack an owner byte-bound actual PASS.
# Do not rerun qualified bank44/wide40/PARTIAL29/F33 bodies.
