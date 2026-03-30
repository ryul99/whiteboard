from mcp.server.fastmcp import FastMCP

mcp = FastMCP("whiteboard")
store: dict[str, str] = {}


@mcp.tool()
def write(key: str, text: str) -> str:
    """Save text under a key."""
    store[key] = text
    return f"Saved to '{key}'."


@mcp.tool()
def read(key: str) -> str:
    """Recall saved text by key."""
    if key not in store:
        return f"No data found for '{key}'."
    return store[key]


if __name__ == "__main__":
    mcp.run()
