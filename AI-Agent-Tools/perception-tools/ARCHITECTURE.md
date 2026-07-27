# Architecture

`perception_tools` is a single installable package with two entry points:

- `perception-tools`: direct CLI discovery and execution
- `perception-tools-mcp`: stdio MCP server

`server.py` registers 53 thin MCP adapters. Domain modules implement search,
filesystem, documents, media, public data, and private data. Functions return a
consistent MCP text response containing serialized `ActionResponse` JSON.

Optional libraries are imported through dependency proxies. This keeps package
and server imports usable with the core installation, while a tool that needs a
missing extra returns an actionable installation error.

Hosted media inference is isolated behind the official OpenAI client and reads
`OPENAI_API_KEY` plus the optional `PERCEPTION_VISION_MODEL`. Public-data tools
call their named upstream services directly. Network tests are marked `live`
and are opt-in.
