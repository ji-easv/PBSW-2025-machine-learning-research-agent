from datetime import datetime
from pathlib import Path
from typing import Literal

from autogen import ChatResult

MAX_INTERNAL_ROUNDS = 10

FINAL_ANSWER_FORMAT = """
FINAL ANSWER FORMAT:
When you have gathered results satisfying all constraints, respond with:
RESULT:
<list of papers in the format:>
1. Title: <title>
   Authors: <authors>
   Year: <publication year>
   Citations: <number of citations>
   URL: <URL>
"""


def get_llm_config(llm_provider: Literal["mistral", "google"], api_key: str):
    if llm_provider == "mistral":
        return {
            "config_list": [
                {
                    "model": "mistral-small-2506",
                    "api_type": "mistral",
                    "api_key": api_key,
                    "api_rate_limit": 0.1,
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
    elif llm_provider == "google":
        return {
            "config_list": [
                {
                    "model": "gemini-2.0-flash",
                    "api_type": "google",
                    "api_key": api_key,
                    "api_rate_limit": 0.1,
                    "max_retries": 3,
                    "num_predict": -1,
                    "repeat_penalty": 1.1,
                    "native_tool_calls": False,
                    "stream": False,
                    "seed": 23,
                    "cache_seed": None,
                    "timeout": 30,
                }
            ]
        }
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def get_work_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    p = Path.cwd() / "coding" / timestamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results(chat: ChatResult, filename: str = "logs/latest_results.md"):
    """
    Extracts the judge evaluation and each agent's final answer from the chat result,
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Extract judge evaluation (JSON block with TERMINATE)
    judge_eval = None
    for msg in reversed(chat.chat_history):
        content = msg.get("content", "")
        if not content or not content.strip():
            continue
        if "{" in content and "}" in content and "TERMINATE:" in content:
            judge_eval = content.replace("TERMINATE:", "").strip()
            break

    # Extract agent results (RESULT from each agent)
    agent_results = {}
    agent_names = ["ResearchPaperAPIAgent", "WebSearchOrchestrator"]
    for agent in agent_names:
        for msg in reversed(chat.chat_history):
            name = msg.get("name", "")
            content = msg.get("content", "")
            if not content or not content.strip():
                continue
            if name != agent:
                continue
            if content.strip().startswith("RESULT:") or content.strip().startswith(
                "RESULT:"
            ):
                agent_results[agent] = content.strip()
                break
            if content.strip().startswith("OK:") and "RESULT:" in content:
                agent_results[agent] = content.strip()
                break

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Agent Results - {timestamp}\n\n")
        if judge_eval:
            f.write("## Judge Evaluation\n\n")
            f.write(f"{judge_eval}\n\n")
        for agent, result in agent_results.items():
            f.write(f"## {agent}\n\n{result}\n\n")
    print(f"Results saved to {filename}")
