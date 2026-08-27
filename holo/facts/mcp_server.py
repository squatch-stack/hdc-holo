"""The facts MCP server: a thin stdio shim over holo.facts.query.

Three tools, Context7-shaped (resolve, then query):

  search_claims(query, status, limit)  ranked registry records
  get_claim(id)                        record + supersession chain +
                                       live derivation + cite sites
  search_kb(query, limit)              matrix search over a knowledge-base
                                       checkout (HOLO_KB_PATH)

Register for Claude Code:  claude mcp add holo-facts -- holo-facts mcp
(run from the repo, or set the server cwd to it — the server resolves
`claims/config.json` upward from its working directory).

The `mcp` dependency is the `facts` extra (`pip install
'hdc-holo[facts]'`); everything else in holo.facts stays stdlib+numpy.
"""

__all__ = ["serve"]


def serve(root):
    # mcp 2.x renamed FastMCP -> MCPServer (and made the old import
    # raise a migration error); same constructor/tool/run surface for
    # our usage, so support both majors
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
        except ImportError as e:
            raise RuntimeError(
                "the MCP server needs the `mcp` package (Python >= "
                "3.10) — pip install 'hdc-holo[facts]' from a 3.10+ "
                "environment; the checker itself stays 3.9-compatible") from e

    from . import query as q

    app = _Server("holo-facts")

    @app.tool()
    def search_claims(query: str, status: str = "current",
                      limit: int = 8) -> dict:
        """Search the repo's registered claims (measured numbers with
        provenance). status: current | superseded | retracted | any."""
        return q.search_claims(root, query, status=status, limit=limit)

    @app.tool()
    def get_claim(id: str) -> dict:
        """Full record for one claim id: rendered statement, value,
        supersession chain, live derivation, and cite sites."""
        return q.get_claim(root, id)

    @app.tool()
    def search_kb(query: str, limit: int = 8) -> dict:
        """Search the configured knowledge base (topics, algorithms,
        learnings; returns arXiv ids per hit)."""
        return q.search_kb(root, query, limit=limit)

    app.run()
