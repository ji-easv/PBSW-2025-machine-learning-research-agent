from datetime import datetime
from pathlib import Path

MAX_INTERNAL_ROUNDS = 5

FINAL_ANSWER_FORMAT = """
FINAL ANSWER FORMAT:
When you have gathered results satisfying all constraints, respond with:
RESULTS:
<list of papers in the format:>
1. Title: <title>
   Authors: <authors>
   Year: <publication year>
   Citations: <number of citations>
   URL: <URL>
"""


def get_llm_config(api_key: str):
    return {
        "config_list": [
            {
                "model": "open-mistral-nemo",
                "api_type": "mistral",
                "api_key": api_key,
                "api_rate_limit": 0.5,
                "max_retries": 3,
                "timeout": 30,
                "num_predict": -1,
                "repeat_penalty": 1.1,
                "stream": False,
                "seed": 42,
                "native_tool_calls": False,
                "cache_seed": None,
            }
        ]
    }


def get_work_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    p = Path.cwd() / "coding" / timestamp
    p.mkdir(parents=True, exist_ok=True)
    return p
