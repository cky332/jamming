from typing import Dict
from openai import OpenAI, AzureOpenAI


def make_client(config: Dict):
    api_type = config.get("api_type", "").lower()
    if api_type == "openai":
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
        raise ValueError(f"Unknown api_type={api_type!r}; expected 'openai' or 'AZURE'")
