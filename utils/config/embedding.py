import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, model_validator

from ...utils.env import resolve_env_vars_in_dict

DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-ada-002"

logger = logging.getLogger()


class UnsupportedProviderError(ValueError):
    """Raised when an unsupported embedding provider type is specified."""

    def __init__(self, provider_type: str) -> None:
        super().__init__(
            f"Unsupported embedding_provider_type: '{provider_type}'. Must be 'sentence_transformers' or 'openai'."
        )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""

    provider_type: str = Field(
        default="openai", description="Type of embedding provider: 'sentence_transformers' or 'openai'"
    )
    model_name: str | None = Field(
        default=None, description="Model name for the selected provider. Can be a placeholder like ${ENV_VAR}."
    )
    # OpenAI specific, optional
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible embeddings API. Can be a placeholder like ${ENV_VAR}.",
    )
    api_key: str | None = Field(default=None, description="API key for provider. Can be a placeholder like ${ENV_VAR}.")

    @model_validator(mode="after")
    def process_and_validate_config(self) -> "EmbeddingConfig":
        """
        Resolves placeholders in string fields and sets defaults after initial model creation.
        """

        fields_to_resolve_placeholders = {}
        if self.model_name is not None:
            fields_to_resolve_placeholders["model_name"] = self.model_name
        if self.base_url is not None:
            fields_to_resolve_placeholders["base_url"] = self.base_url
        if self.api_key is not None:
            fields_to_resolve_placeholders["api_key"] = self.api_key

        if fields_to_resolve_placeholders:
            resolved_fields = resolve_env_vars_in_dict(fields_to_resolve_placeholders)
            self.model_name = resolved_fields.get("model_name", self.model_name)
            self.base_url = resolved_fields.get("base_url", self.base_url)
            self.api_key = resolved_fields.get("api_key", self.api_key)

        provider_type = self.provider_type
        model_name = self.model_name
        base_url = self.base_url
        api_key = self.api_key

        if provider_type == "openai":
            if model_name is None:
                self.model_name = DEFAULT_OPENAI_MODEL
                logger.info(f"embedding.model_name not set for OpenAI, defaulting to '{self.model_name}'.")
            if api_key is None:
                logger.error("embedding.api_key is None for OpenAI provider. Langchain might try 'OPENAI_API_KEY'.")

        elif provider_type == "sentence_transformers":
            if model_name is None:
                self.model_name = DEFAULT_SENTENCE_TRANSFORMERS_MODEL
                logger.info(
                    f"embedding.model_name not set for SentenceTransformers, defaulting to '{self.model_name}'."
                )
            if base_url is not None:
                logger.warning(
                    f"embedding.base_url ('{base_url}') is set but will be ignored for sentence_transformers provider."
                )
            if api_key is not None:
                logger.warning("embedding.api_key is set but will be ignored for sentence_transformers provider.")

        else:
            raise UnsupportedProviderError(provider_type)

        return self


def create_embedding_function(config: EmbeddingConfig) -> Embeddings:
    if config.provider_type == "openai":
        logger.info("Using OpenAIEmbeddings", model=config.model_name, base_url=config.base_url)

        return OpenAIEmbeddings(model=config.model_name, base_url=config.base_url, api_key=config.api_key)
    elif config.provider_type == "sentence_transformers":
        logger.info("Using SentenceTransformerEmbeddings", model=config.model_name)

        return HuggingFaceEmbeddings(model_name=config.model_name)
    else:
        UnsupportedProviderError(config.pov)
