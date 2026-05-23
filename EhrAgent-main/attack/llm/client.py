import os
from typing import List, Dict, Optional
from openai import OpenAI

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"


class DeepSeekClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = SILICONFLOW_BASE_URL,
        api_key_env: str = "SILICONFLOW_API_KEY",
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set. "
                f"v0.1 requires SiliconFlow API key for DeepSeek-V3.2."
            )
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 800,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        if temperature != 0.0:
            raise ValueError(
                "v0.1 pins temperature=0 everywhere for reproducibility. "
                "Override blocked."
            )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


def deepseek_v32_config(seed: Optional[int] = None) -> Dict:
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is required")
    return {
        "model": DEFAULT_MODEL,
        "api_key": api_key,
        "base_url": SILICONFLOW_BASE_URL,
        "api_type": "openai",
    }
