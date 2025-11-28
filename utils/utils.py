from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

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

If you cannot find results satisfying all constraints after multiple attempts, respond with:
JUSTIFICATION:
<brief explanation of why the constraints could not be met>
RESULT:
[]
"""

EVALUATION_CRITERIA = """
When evaluating, consider:
    - completness (1-5): Did the agent satisfy all explicit constraints in the task (e.g., publication year, citation count, number of results)?
    - relevance (1-5): Are the returned papers relevant to the requested topic?
    - honesty & transparency (1-5): Did the agent avoid fabricating citation counts or details, and did it explain any limitations of the tools used?
    - clarity & structure (1-5): Is the answer easy to read, with titles, authors, years, citation counts, and URLs clearly listed where available?
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


def extract_final_answer(chat: ChatResult, agent_name: str) -> str:
    """
    Extracts the final answer (RESULT:) from the agent's chat history.
    """

    for msg in reversed(chat.chat_history):
        name = msg.get("name", "")
        content = msg.get("content", "")
        if not content or not content.strip():
            continue
        if name != agent_name:
            continue
        if "RESULT:" in content.strip():
            # Extract everything after "RESULT:"
            content = content.replace("TERMINATE:", "")
            result_index = content.index("RESULT:") + len("RESULT:")
            return content[result_index:].strip()
    return "No RESULT found."


def save_results(
    chat: ChatResult, judge_eval: Dict, filename: str = "latest_results.md"
):
    """
    Extracts the judge evaluation and each agent's final answer from the chat result,
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Prepend timestamp to filename and save under logs/
    base_filename = filename if filename.endswith(".md") else f"{filename}.md"
    log_filename = logs_dir / f"{timestamp}_{base_filename}"

    # Extract agent results (RESULT from each agent)
    agent_results = {}
    agent_names = ["ResearchPaperAPIAgent", "WebSearchOrchestrator"]
    for agent in agent_names:
        agent_results[agent] = extract_final_answer(chat, agent)

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Agent Results - {timestamp}\n\n")
        if judge_eval:
            f.write("## Judge Evaluation\n\n")
            f.write(f"{judge_eval}\n\n")
        for agent, result in agent_results.items():
            f.write(f"## {agent}\n\n{result}\n\n")
    print(f"Results saved to {log_filename}")
