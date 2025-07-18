from typing import Any

from pydantic import BaseModel, Field, model_validator


class InvalidTransportError(ValueError):
    """Raised when an invalid transport type is specified."""

    def __init__(self, transport_type: str) -> None:
        super().__init__(f"Invalid transport_type: {transport_type}. Must be 'stdio' or 'streamable-http'.")


class MissingURLError(ValueError):
    """Raised when URL is missing for streamable-http transport."""

    def __init__(self) -> None:
        super().__init__("URL must be provided for 'streamable-http' transport type.")


class MissingCommandError(ValueError):
    """Raised when command is missing for stdio transport."""

    def __init__(self) -> None:
        super().__init__("Command must be provided for 'stdio' transport type.")


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    transport_type: str = Field(default="stdio", description="Transport type: 'stdio' or 'streamable-http'")

    # Stdio specific fields
    command: str | None = Field(default=None, description="Command to start the MCP server (for stdio)")
    args: list[str] | None = Field(default=None, description="Arguments for the command (for stdio)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for the command (for stdio)")

    # Streamable-HTTP specific fields
    url: str | None = Field(default=None, description="URL of the MCP server (for http transport)")
    headers: dict[str, str] | None = Field(
        default=None, description="HTTP headers for the MCP server (for http transport)"
    )

    description: str
    tags: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)  # This might be populated dynamically

    @model_validator(mode="after")
    def check_transport_requirements(cls, values: Any) -> Any:  # noqa: N805
        transport_type = values.transport_type
        url = values.url
        command = values.command

        if transport_type == "streamable-http" and not url:
            raise MissingURLError()

        if transport_type == "stdio" and not command:
            raise MissingCommandError()

        if transport_type not in ["stdio", "streamable-http"]:
            raise InvalidTransportError(transport_type)

        return values

    @property
    def mcp_command(self) -> str:
        """Get the command for MCP client. Relevant for stdio."""
        if self.transport_type == "stdio" and self.command:
            return self.command
        return ""

    @property
    def mcp_args(self) -> list[str]:
        """Get the args for MCP client. Relevant for stdio."""
        if self.transport_type == "stdio" and self.args is not None:
            return self.args
        return []
