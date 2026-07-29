"""
Event Server - compatibility shim.

This module used to be a second, near-identical copy of server.py. Keeping two
copies in sync failed in practice: fixes landed in one and not the other, so
whichever entry point you happened to run decided which bugs you got.

Everything now lives in server.py and is re-exported here, so existing
`python server_fastapi.py` invocations and `import server_fastapi` keep
working while there is only one implementation to maintain.
"""

from server import (  # noqa: F401  (re-exported for backwards compatibility)
    _env_int,
    agent,
    agent_lock,
    app,
    build_parser,
    init_agent,
    load_mcp_tools_async,
    main,
    mcp_loading_status,
    EventRequest,
    ProcessRegister,
    ProcessUnregister,
)

if __name__ == "__main__":
    main()
