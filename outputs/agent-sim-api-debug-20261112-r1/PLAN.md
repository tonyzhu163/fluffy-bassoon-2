# API harness qualification

Created November 12, 2026 from main `42d47d5c`. The user authorized debugging
the agent simulator with their existing OpenAI API key, a new Baige terminal,
and Git deployment. Credentials stay outside Git and evidence.

The direct Responses client uses the existing MCP tool definitions and images,
executes tools serially, records API usage, and stops on uncertain or failed
calls. Qualify this transport before collecting agent demonstrations. A passing
smoke does not establish manipulation success or a useful teaching trajectory.

1. Check the configured credential and exact `gpt-6-astra` model. Completed:
   HTTP 200, response `READY`, 17 input and 5 output tokens, 5.43 seconds.
   This network connectivity check performed no local model computation.
2. Deploy a committed client through Git. Use an incremental Git bundle only
   if the private remote cannot be fetched from the allocated host.
3. Run the client against the existing synthetic MCP fixture on allocated
   Linux to verify image outputs, serial calls, finish, and usage records.
4. Qualify the allocated GPU and native RoboCasa environment, then repeat the
   short tool sequence on a native episode. Inspect executed-action and image
   records. Record any runtime mismatch before attempting a fresh output ID.

Initial bound: at most eight API calls per smoke, 1,024 output tokens per call,
and two physical controls per episode. At most one GPU-hour for qualification;
record actual usage. No learner fitting or demonstration collection campaign.

Use fresh output identities for every attempt. Existing jobs, source worktrees,
shared environments and model caches remain immutable.

## Execution update

The live API/MCP fixture passed on Baige at `1ad1afd4`: five provider calls,
three image-bearing results, two recorded controls and one exported example.
Direct Baige provider access timed out. A temporary loopback CONNECT relay
restricted to `api.openai.com:443` and an authenticated SSH reverse forward
provided end-to-end TLS access; the credential was passed in memory via SSH
stdin. No credential was written on Baige or supplied to the simulator.

The separately owned [native demonstration](../astra-robocasa-demo-20261112-r1/PLAN.md)
completed and its GPU process exited. Reuse that qualified runtime unchanged
for this client's two-control native smoke, with a new output directory and
the latest committed source. This is still interface qualification, not an
additional task-solving attempt. A lightweight CUDA check and real episode
qualify this evaluation-only reuse; record its conservative GPU window.
