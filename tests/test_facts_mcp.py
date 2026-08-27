"""End-to-end handshake with the facts MCP server over real stdio.

Dedicated file because `mcp` is the optional `facts` extra and needs
Python >= 3.10 (importorskip at file scope per tests/TESTING.md); the
tool LOGIC is tested stdlib-only in test_claims.py — this file proves
the wire: spawn `holo-facts mcp`, initialize, list tools, call one.
CI runs it on the 3.12 leg with `.[facts]` installed.
"""

import pytest

mcp = pytest.importorskip("mcp")

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_stdio_server_lists_tools_and_answers():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "holo.facts.cli", "mcp"], cwd=ROOT)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools = sorted(t.name for t in (await s.list_tools()).tools)
            assert tools == ["get_claim", "search_claims", "search_kb"]
            res = await s.call_tool(
                "search_claims", {"query": "encode kernel speedup"})
            payload = getattr(res, "structuredContent", None) \
                    or json.loads(res.content[0].text)
            if "results" not in payload:   # some SDKs wrap: {"result": ...}
                payload = payload.get("result", payload)
            assert payload["results"][0]["id"] == "accel.encode_speedup"

    asyncio.run(run())
