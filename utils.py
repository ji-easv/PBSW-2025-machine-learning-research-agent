from datetime import datetime
from pathlib import Path


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


ReAct_prompt = """
To solve problems, you will use the ReAct (Reasoning and Acting) framework.

Process:
1. Thought: Reason about what to do next
2. [Call the appropriate tool - AutoGen handles this automatically]
3. [Receive tool results from the system]
4. Thought: Analyze the results and decide next steps
5. Repeat steps 1-4 if needed
6. Final Answer: Provide the answer when task is complete. End your response with 'TERMINATE'.
"""
