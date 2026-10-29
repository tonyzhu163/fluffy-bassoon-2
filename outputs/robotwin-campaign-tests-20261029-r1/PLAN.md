# Allocated campaign integration tests

2026-10-29 Asia/Shanghai. Validate the new one-million-action campaign ledger,
raw corpus and fresh-PPO metadata integration before any larger training slice.
Run all9focused test files on allocated Linux. The tests include conservative
crash accounting, exact budget cuts, raw array/hash round trips, native feedback
preservation, behavior metadata, and interrupted driver recovery that skips
completed source-validation/native/target units. No simulator is instantiated
by this job; it generates no robot-control actions and earns no source credit.

One Baige GPU,16CPUs,128GiB RAM,8GiB shared memory, **4-minute cap**;180second
pytest timeout plus15second termination grace. The prior measured scheduler
durations plus the conservative r1reserve total41minutes. This final check
therefore remains within the initial45GPU-minute resource envelope. Record
the actual allocation probe, full test result, source commit and terminal time.

The two-hour actual training continuation remains a separate resource decision.
