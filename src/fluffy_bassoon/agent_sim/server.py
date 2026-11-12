"""The same stdio MCP server works with Codex and Claude Code.

Simulator calls run serially on the main thread (EGL/GL contexts are thread
affine). Stdout is reserved at the file-descriptor level for the MCP protocol;
Python and native simulator logging go to stderr.
"""
import argparse
import base64
import importlib
import json
import os
from pathlib import Path
import sys

from .core import Session, encode


def tools_schema():
    from mcp.types import Tool, ToolAnnotations

    def tool(name, description, properties, required, readonly=False):
        return Tool(name=name, description=description,
            inputSchema=dict(type="object", properties=properties, required=required, additionalProperties=False),
            annotations=ToolAnnotations(readOnlyHint=readonly, destructiveHint=False,
                                        idempotentHint=readonly, openWorldHint=False))

    text = dict(type="string", minLength=1)
    return [
        tool("sim_interface", "Read the native action convention, observation fields and collection protocol.", {}, [], True),
        tool("sim_reset", "Start a fresh seeded episode. Finish the previous episode first; IDs cannot be reused.",
             dict(task=text, seed=dict(type="integer", minimum=0, maximum=2**32-1), episode_id=text),
             ["task", "seed", "episode_id"]),
        tool("sim_observe", "Read current RGB views and proprioception without advancing physics.", {}, [], True),
        tool("sim_execute", "Execute a short native action sequence. First state your subgoal, concise public "
             "decision summary, and expected visible outcome. These are training annotations, not private reasoning. "
             "Use observation_control from the latest observation. Physics pauses between calls.",
             dict(actions=dict(type="array", minItems=1, items=dict(type="array", minItems=1,
                       items=dict(type="number"))), observation_control=dict(type="integer", minimum=0),
                  subgoal=text, summary=text, expected_outcome=text),
             ["actions", "observation_control", "subgoal", "summary", "expected_outcome"]),
        tool("sim_finish", "Finish the episode with a concise outcome/recovery summary and your own success claim. "
             "Your claim never replaces the simulator's separately recorded native outcome.",
             dict(summary=text, success_claim=dict(type="boolean")), ["summary", "success_claim"]),
    ]


def call(session, name, arguments):
    if name == "sim_interface":
        return dict(interface=session.backend.describe(), protocol=session.manifest)
    if name == "sim_reset":
        return session.reset(**arguments)
    if name == "sim_observe":
        return session.observe()
    if name == "sim_execute":
        return session.execute(**arguments)
    if name == "sim_finish":
        return session.finish(**arguments)
    raise ValueError("Unknown simulator tool")


def content(session, result):
    from mcp.types import ImageContent, TextContent

    blocks = [TextContent(type="text", text=encode(result))]
    for name, ref in result.get("observation", {}).get("images", {}).items():
        blocks.append(TextContent(type="text", text="Camera: " + name))
        blocks.append(ImageContent(type="image", mimeType="image/png",
            data=base64.b64encode((session.output / ref["path"]).read_bytes()).decode()))
    return blocks


async def serve(session, protocol_stdout):
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, TextContent

    server = Server("fluffy-bassoon-simulator")

    @server.list_tools()
    async def list_tools():
        return tools_schema()

    @server.call_tool()
    async def call_tool(name, arguments):
        # Deliberately no await inside native execution: cancellation cannot
        # release the episode to a second physical dispatch mid-action.
        try:
            return CallToolResult(content=content(session, call(session, name, arguments)))
        except Exception as error:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=str(error))])

    async with stdio_server(stdout=anyio.wrap_file(protocol_stdout)) as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh collection directory; never reused")
    parser.add_argument("--client", choices=("codex", "claude", "other"), required=True)
    parser.add_argument("--model", required=True, help="Declared model identity; MCP cannot verify client selection")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-controls", type=int, required=True, help="Explicit episode control budget")
    parser.add_argument("--max-chunk", type=int, default=50)
    parser.add_argument("--feedback", choices=("hidden", "native"), default="hidden")
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--execution", choices=("baige", "slurm", "direct"), default="baige")
    parser.add_argument("--backend-factory", help="Trusted local module:callable implementing core.Backend")
    args = parser.parse_args(argv)
    # Save the original pipe before redirecting fd 1, including C/C++ output.
    sys.stdout.flush()
    protocol_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    import anyio

    if args.backend_factory:
        module, name = args.backend_factory.split(":", 1)
        backend = getattr(importlib.import_module(module), name)()
    else:
        from .robocasa import RoboCasaBackend
        backend = RoboCasaBackend(split=args.split, execution=args.execution)
    session = Session(backend, args.output, client=args.client, model=args.model,
        source_commit=args.source_commit, max_controls=args.max_controls,
        max_chunk=args.max_chunk, feedback=args.feedback)
    try:
        anyio.run(serve, session, protocol_stdout)
    finally:
        session.close()
        protocol_stdout.close()


if __name__ == "__main__":
    main()
