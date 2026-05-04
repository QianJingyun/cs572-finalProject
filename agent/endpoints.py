"""Load Azure OpenAI endpoint configurations from .env."""
import os
from dataclasses import dataclass


@dataclass
class AzureEndpoint:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str


def load_endpoints(model_suffix: str = "GPT41") -> list[AzureEndpoint]:
    """Load numbered Azure endpoints (AZURE_API_KEY_{i}_{suffix}, etc.) from env.

    Falls back to legacy AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT if no
    numbered endpoints are found.
    """
    api_version = os.environ.get(
        f"AZURE_API_VERSION_{model_suffix}",
        os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
    )
    endpoints = []
    for i in range(1, 10):
        key = os.environ.get(f"AZURE_API_KEY_{i}_{model_suffix}")
        base = os.environ.get(f"AZURE_API_BASE_{i}_{model_suffix}")
        deployment = os.environ.get(f"AZURE_DEPLOYMENT_{i}_{model_suffix}", "gpt-4.1")
        if key and base:
            endpoints.append(AzureEndpoint(
                api_key=key, endpoint=base,
                deployment=deployment, api_version=api_version,
            ))

    if not endpoints:
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if key and base:
            endpoints.append(AzureEndpoint(
                api_key=key, endpoint=base,
                deployment="gpt-4.1", api_version=api_version,
            ))

    return endpoints
