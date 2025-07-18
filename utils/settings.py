import asyncio
from pathlib import Path

import logging
import yaml
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

import os
import re

from .config.mcp_server import MCPServerConfig

logger = logging.getLogger()


class Settings(BaseSettings):
    """Global application settings."""

    # Paths
    config_dir: Path = Path("config")
    cache_dir: Path = Path(".cache")
    log_dir: Path = Path("logs")
    mcp_servers_file: Path = Path("config/mcp_servers.yaml")
    models_file: Path = Path("config/models.yaml")

    # Vector store
    vector_store_path: Path = Path(".cache/chroma_db")

    # MCP settings
    mcp_timeout: float = 30.0
    mcp_max_retries: int = 3
    mcp_process_pool_size: int = 5

    # Security
    mask_tokens: bool = True
    token_pattern: str = r"(YC[a-zA-Z0-9_-]{32,}|AQVN[a-zA-Z0-9_-]{32,})"

    # Performance
    rag_top_k: int = 8
    max_context_tokens: int = 8192
    response_timeout: float = 60.0

    # Logging
    console_logging_enabled: bool = True

    model_config = ConfigDict(env_prefix="META_MCP_")


def _load_mcp_servers_sync(config_path: Path | None = None) -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from YAML file (sync version)."""
    if config_path is None:
        settings = Settings()
        config_path = settings.config_dir / "mcp_servers.yaml"

    with config_path.open() as f:
        data = yaml.safe_load(f)

    return {name: MCPServerConfig(**config) for name, config in data.items()}


def load_mcp_servers_config(config_path: Path | None = None) -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from YAML file."""
    return _load_mcp_servers_sync(config_path)


async def load_mcp_servers_async(config_path: Path | None = None) -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from YAML file asynchronously."""
    return await asyncio.to_thread(_load_mcp_servers_sync, config_path)

ENV_VAR_PATTERN = re.compile(r"\$\{(?P<var_name>[A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_env_vars_in_dict(input_data: dict[str, str]) -> dict[str, str]:
    """
    Resolves environment variable placeholders (e.g., ${VAR_NAME})
    in the string values of a dictionary.
    """
    if not input_data:
        return {}

    resolved_data = {}
    for key, value in input_data.items():
        if isinstance(value, str):
            match = ENV_VAR_PATTERN.fullmatch(value)
            if match:
                var_name = match.group("var_name")
                env_value = os.environ.get(var_name)
                if env_value is not None:
                    resolved_data[key] = env_value
                else:
                    logger.warning(
                        "Environment variable not found for config key, using original value.",
                        key=key,
                        placeholder=value,
                        variable_name=var_name,
                    )
                    resolved_data[key] = value
            else:
                resolved_data[key] = value
        else:
            resolved_data[key] = value
    return resolved_data


def resolve_env_var(value: str) -> str:
    """
    Resolves a single environment variable placeholder (e.g., ${VAR_NAME}).
    """
    if not isinstance(value, str):
        return value

    # Check if the value contains ${...} pattern
    if "${" in value and "}" in value:
        # Replace all occurrences of ${VAR_NAME} with their values
        import re
        def replacer(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                logger.info(f"Resolved env var {var_name} to {env_value}")
                return env_value
            else:
                logger.warning(f"Environment variable {var_name} not found")
                return match.group(0)  # Return original if not found

        pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}'
        result = re.sub(pattern, replacer, value)
        return result

    return value