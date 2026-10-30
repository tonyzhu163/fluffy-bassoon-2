# Four-way LIBERO90 noise diagnostic

Frozen predeclared grid: all90 task IDs, state0/seed123; native independent Gaussian, zero, tied unbounded Gaussian, tied clipped Gaussian. No learned controllers or checkpoint selection.

360 query episodes;147600 worst-case simulator actions including settling,152000 joint cap. Two workers,3600 cumulative GPU-seconds and76000 actions each; retries/setup/shutdown consume the same holds. Funded from released terminal time within original32GPU-hours, not new compute. The source collection and completed first-wave ledgers stay unchanged.

Source is pinned in plan.json and target-ledger.json. Linux focused tests, allocatedGPU probe and actual first planned reset qualification precede rollout. Complete raw observations/actions/noises/rewards/boundaries/RNG/cursors and conservative RPC journals are retained on Baige.

rejected-before-launch-wrong-source.json is archival metadata from an unlaunched plan containing a mistyped source; it consumed no simulator actions or GPU time. It is not admitted by the launcher.

Launched22:08Shanghai: worker0 `job-g8vas173mmy5`; worker1 `job-834arjyjj6mp`. Source `53b6c0d1bcccd6d94025762335b66a72f09a3119`, plan-fileSHA `9f5f86d6bffec1922952cf0da8e3dc657beccd361db2b0fda132837c5e9e0c7d`, ledgerSHA `0de51e4e6fec4c4929cc801f64ea043bc530f115fc68ff2da989d0ed2571b4c2`. ActualGPU/test/episode verification remains required before interpreting results.
