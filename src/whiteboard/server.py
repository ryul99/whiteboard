from mcp.server.fastmcp import FastMCP

mcp = FastMCP("whiteboard")
store: dict[str, str] = {}


@mcp.tool()
def write(key: str, text: str) -> str:
    """
    Write the thoughts at note and name the note as `key`
    """
    store[key] = text
    return f"Saved to '{key}'."


@mcp.tool()
def read(key: str) -> str:
    """
    Recall the note named as `key`
    """
    if key not in store:
        return f"No data found for '{key}'."
    return store[key]


if __name__ == "__main__":
    mcp.run()
