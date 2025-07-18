import threading

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import BaseTool

from settings import resolve_env_var, resolve_env_vars_in_dict
from settings import Settings, load_mcp_servers_config

class MCPFacade:
    def __init__(self, settings: Settings):
        mcp_servers_config = load_mcp_servers_config(settings.mcp_servers_file)

        self.available_mcp_servers_ids = mcp_servers_config.keys()

        langchain_compatible_config = {}
        for server_id, server_config in mcp_servers_config.items():
            if server_config.transport_type == "stdio":
                resolved_command = resolve_env_var(server_config.command)
                langchain_compatible_config[server_id] = {
                    "command": resolved_command,
                    "args": server_config.args,
                    "env": resolve_env_vars_in_dict(server_config.env),
                    "transport": "stdio",
                }
            elif server_config.transport_type == "streamable-http":
                langchain_compatible_config[server_id] = {
                    "url": server_config.url,
                    "headers": resolve_env_vars_in_dict(server_config.headers),
                    "transport": "streamable_http",
                }

        self.mcp_client = MultiServerMCPClient(langchain_compatible_config)

        self.loaded_tools: dict[str, list[BaseTool]] = {}
        self._tools_lock = threading.Lock()

    async def __locked_load_tools(self, server_id: str) -> list[BaseTool]:
        with self._tools_lock:
            if server_id in self.loaded_tools:
                return self.loaded_tools[server_id]
            else:
                tools = await self.mcp_client.get_tools(server_name=server_id)

                patched_tools = []
                for tool in tools:
                    if tool.metadata is None:
                        tool.metadata = {}
                    tool.metadata["server_name"] = server_id

                    patched_tools.append(tool)

                self.loaded_tools[server_id] = patched_tools

                return patched_tools

    async def list_tools(self, server_id: str | None = None) -> list[BaseTool]:
        if server_id is None:
            result = []

            for sid in self.available_mcp_servers_ids:
                result.extend(await self.__locked_load_tools(sid))

            return result

        return await self.__locked_load_tools(server_id)

    async def get_tool(self, server_name: str, tool_name: str) -> BaseTool | None:
        tools = await self.__locked_load_tools(server_name)

        return next((tool for tool in tools if tool.name == tool_name), None)
