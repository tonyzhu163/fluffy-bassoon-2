# Direct Astra API and native simulator qualification

The bounded Responses API client passed both the existing synthetic MCP fixture
and native RoboCasa365 on allocated Baige Linux. The native check used the
unchanged runtime already qualified by the separately owned
[successful Astra demonstration](../astra-robocasa-demo-20261112-r1/RESULT.md).
No learner training or demonstration collection campaign was launched.

| Measurement | Synthetic fixture | Native OpenDrawer, pretrain, seed 7 |
| --- | --- | --- |
| Actual provider model | `gpt-6-astra` | `gpt-6-astra` |
| Source commit | `1ad1afd4a62dcc1068688bcf8d118226339f7fd7` | `c86c1d3fe02e0b511841bd52dfec96cc8c37e9e8` |
| API calls | 5 | 5 |
| Input / output tokens | 5,878 / 246 | 13,759 / 265 |
| Client wall time | 22.52 seconds | 41.09 seconds |
| Executed controls / decisions | 2 / 1 | 2 / 1 |
| Image-bearing tool results | 3, one view each | 3, three views each |
| Exported causal examples | 1 | 1 |

Both loops completed `sim_interface → sim_reset → sim_execute → sim_observe →
sim_finish`, with no tool errors. Native export verified two 12-dimensional
zero-action vectors, three consecutive observations, and all nine camera PNG
hashes. The two-control episode budget caused truncation as intended. Native
success and the agent's success claim were both false: this was a wiring check,
not an attempt to open the drawer. The separate 220-control successful
demonstration must not be conflated with this API smoke.

The native GPU window, including interpreter/CUDA startup and waiting for API
calls, was 57.96 seconds or **0.016101 GPU-hours**, below the 1 GPU-hour ceiling.
The synthetic fixture used CPU only. Post-run inspection found no GPU compute
process and a clean detached source checkout. The temporary relay and SSH
reverse forward were stopped. The existing allocated instance remains running.

## Changes and resolved failures

- Added `fluffy_bassoon.agent_sim.api_client`: serial Responses function calls
  from the MCP server's own schemas, image tool outputs, in-memory reasoning
  continuity, explicit call/token/time budgets, public context/usage records,
  and stop-on-error behavior without replaying uncertain physical calls.
- Direct provider access from Baige timed out. A temporary loopback CONNECT
  relay restricted to `api.openai.com:443`, reached over authenticated SSH,
  carried end-to-end TLS with ordinary certificate verification. This is a
  connectivity workaround, not a permanent deployed service.
- Deployed committed code through Git. When the remote fetch stalled, used
  small incremental Git bundles and verified exact detached commits. No source
  folder was copied. An initial fixture invocation failed before any API call
  because the sparse checkout had not materialized; explicit Git checkout
  repaired that, and the passing attempt used a fresh output identity.
- The credential came from the user's local file through SSH stdin into the
  API process's memory/environment. It was not written on Baige, included in
  command arguments, passed to the simulator subprocess, or committed. Exact
  credential-byte checks passed for the retained evidence.

Mac validation was syntax-only. Live provider calls, simulator execution and
export checks ran on Linux. These checks establish tool/image transport and
aligned recording; they do not establish task-general teaching quality,
cross-scene transfer, one-shot absorption, or a benefit from TTT layers.

## Evidence

[Fixture checks](VALIDATION.json), [fixture API result](client-result.json),
[fixture API events](client-events.jsonl), [native checks](NATIVE-VALIDATION.json),
[native execution and GPU accounting](NATIVE-EXECUTION.json),
[native CUDA check](native-gpu-light.json),
[native API result](native-client-result.json),
[native API events](native-client-events.jsonl),
[native episode outcome](native-episode-result.json), and
[native export manifest](native-export-manifest.json).

Full journals, PNGs and numeric exports remain under
`/mowenjie/fluffy-bassoon-2/outputs/agent-sim-api-debug-20261112-r1/fixture02`
and `native01` on Baige. Export image references resolve against their recorded
remote session roots; the small manifests committed here do not bundle images.
