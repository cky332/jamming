from typing import Dict
from openai import OpenAI, AzureOpenAI


def make_client(config: Dict):
    # Default to OpenAI when api_type is missing/empty. We omit api_type from
    # SiliconFlow configs to avoid autogen 0.2.0's leak-to-create() quirk
    # (it only strips api_type that startswith("azure")); see config.py.
    api_type = config.get("api_type", "openai").lower()
    if api_type in ("openai", "open_ai", ""):
        return OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
    elif api_type == "azure":
        return AzureOpenAI(
            api_key=config["api_key"],
            azure_endpoint=config["base_url"],
            api_version=config["api_version"],
        )
    else:
        raise ValueError(f"Unknown api_type={api_type!r}; expected 'openai' or 'azure'")
