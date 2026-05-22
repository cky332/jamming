import os


def openai_config(model):
    if model == '<YOUR_OWN_GPT_MODEL_I>':
        config = {
            "model": "<MODEL_NAME>",
            "api_key": "<API_KEY>",
            "base_url": "<BASE_URL>",
            "api_version": "<API_VERSION>",
            "api_type": "AZURE"
        }
    elif model == '<YOUR_OWN_GPT_MODEL_II>':
        config = {
            "model": "<MODEL_NAME>",
            "api_key": "<API_KEY>",
            "base_url": "<BASE_URL>",
            "api_version": "<API_VERSION>",
            "api_type": "AZURE"
        }
    elif model == 'deepseek_v32':
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SILICONFLOW_API_KEY env var required for deepseek_v32 branch"
            )
        config = {
            "model": "deepseek-ai/DeepSeek-V3.2-Exp",
            "api_key": api_key,
            "base_url": "https://api.siliconflow.cn/v1",
            "api_type": "openai",
        }
    else:
        raise ValueError(f"Unknown model: {model!r}")
    return config


def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list
