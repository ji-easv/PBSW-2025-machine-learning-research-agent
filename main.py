import os
import sys
import dotenv
import json
from datetime import datetime
from pathlib import Path
from agents.research_paper_agent import get_research_paper_api_assistant
from agents.user_proxy_agent import get_user_proxy
from agents.web_search_agent import get_web_search_assistant
from autogen.coding import DockerCommandLineCodeExecutor

import logging

from tools.web_search_tool import search_web

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=logging.INFO)

# Set UTF-8 encoding for Windows console to handle Unicode characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
from autogen import (
    AssistantAgent,
)

from utils import (
    get_llm_config,
    get_work_dir,
)

dotenv.load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")

LLM_CONFIG = get_llm_config(api_key=api_key)

executor = DockerCommandLineCodeExecutor(
    work_dir=get_work_dir(),
)

web_search_assistant = get_web_search_assistant(api_key=api_key)
research_paper_api_assistant = get_research_paper_api_assistant(api_key=api_key)
user_proxy = get_user_proxy(executor=executor)


judge = AssistantAgent(
    name="judge",
    llm_config=LLM_CONFIG,
    system_message=(
        "You will be given two sets of results from different agents attempting to "
        "complete the same research task. "
        "Your goals are to: (1) assess how well each agent satisfied the explicit "
        "task constraints (topic, year, citation count, number of results), "
        "(2) evaluate the relevance and clarity of the returned papers, and "
        "(3) reward honesty and clear explanation when no exact solution exists.\n\n"
        "When evaluating, consider:\n"
        "- Constraint satisfaction: Did the agent respect requirements like "
        "'published after 2003' and 'over 10 citations'? Did it clearly say when "
        "no papers met all constraints?\n"
        "- Relevance: Are the returned papers clearly about the requested topic?\n"
        "- Honesty & transparency: Did the agent avoid fabricating citation counts "
        "or details, and did it explain any limitations of the tools used?\n"
        "- Clarity & structure: Is the answer easy to read, with titles, authors, "
        "years, citation counts, and URLs clearly listed where available?\n\n"
        "A good answer may sometimes honestly report that no paper satisfies all "
        "constraints, while providing the best available near-miss papers and "
        "explicitly stating which constraints they fail. Prefer such an answer over "
        "one that ignores constraints or makes up data.\n\n"
        "In your response, briefly compare the two agents, list key strengths and "
        "weaknesses for each, and then clearly state which agent performed better: "
        "'Winner: ResearchPaperAPIAssistant', 'Winner: WebSearchAssistant', or "
        "'Winner: tie' if performance is truly equal. End your response with "
        "'TERMINATE' to indicate the end of the evaluation."
    ),
)


def extract_result_content(chat_result) -> str:
    """Extract actual content from chat result object."""
    if hasattr(chat_result, "chat_history") and chat_result.chat_history:
        # Look for the last assistant message that contains actual results
        # Work backwards through chat history
        for msg in reversed(chat_result.chat_history):
            # Only check messages from the assistant or ResearchPaperAPIAssistant or WebSearchAssistant
            if (
                msg.get("name") in ["ResearchPaperAPIAssistant", "WebSearchAssistant"]
                or msg.get("role") == "assistant"
            ):
                content = msg.get("content", "")
                if not content or not content.strip():
                    continue

                # Skip tool calls and tool responses
                if "***** Suggested tool call" in content:
                    continue
                if "***** Response from calling tool" in content:
                    continue

                # Skip standalone TERMINATE or empty thought messages
                if content.strip() == "TERMINATE":
                    continue
                if (
                    content.strip().startswith("Thought:")
                    and len(content.strip()) < 100
                ):
                    continue

                # This is a real response - clean up TERMINATE if present
                if "TERMINATE" in content:
                    content = content.replace("TERMINATE", "").rstrip()

                return content.strip()

    # Fallback: return summary or string representation
    if hasattr(chat_result, "summary"):
        return chat_result.summary
    return "No results extracted from chat."


def main():
    task = (
        "Find research papers on software testing that satisfy ALL of the following constraints:\n"
        "1) Published after 2003 (i.e., year >= 2004).\n"
        "2) Have more than 10 citations.\n"
        "3) Return the top three articles, providing for each: title, authors, "
        "publication year, number of citations, and URL."
    )

    logging.info("Starting evaluation for task:\n%s", task)

    # Run research paper API assistant
    logging.info("Starting Research Paper API Assistant...")
    paper_result = user_proxy.initiate_chat(
        research_paper_api_assistant,
        message=f"Task: {task}",
        max_turns=10,
    )
    paper_content = extract_result_content(paper_result)

    # Run web search assistant
    logging.info("Starting Web Search Assistant...")
    web_result = user_proxy.initiate_chat(
        web_search_assistant, message=f"Task: {task}", max_turns=10
    )
    web_content = extract_result_content(web_result)

    # Judge evaluates both results
    logging.info("Starting Judge Evaluation...")
    judge_result = user_proxy.initiate_chat(
        judge,
        message=f"""Evaluate the performance of the research agents:

Task: {task}

Research Paper API Assistant Results:
{paper_content}

Web Search Assistant Results:
{web_content}

Provide your evaluation and pick the best results.""",
        max_turns=1,
    )
    judge_content = extract_result_content(judge_result)

    # Save all results to file
    save_results_to_file(task, paper_content, web_content, judge_content)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


def save_results_to_file(
    task: str, paper_content: str, web_content: str, judge_content: str
):
    """Save results to a cleanly formatted markdown file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "latest_results.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# Research Agent Results
Last updated: {timestamp}

---

## Task
{task}
---

## Research Paper API Assistant Results

{paper_content}

---

## Web Search Assistant Results

{web_content}

---

## Judge Evaluation

{judge_content}

---

*This file is automatically overwritten each time the system runs.*
"""

    results_file.write_text(content, encoding="utf-8")
    logging.info("Results saved to: %s", results_file.absolute())

    # Also save a JSON version for programmatic access
    json_file = results_dir / "latest_results.json"
    json_data = {
        "timestamp": timestamp,
        "task": task,
        "paper_api_results": paper_content,
        "web_search_results": web_content,
        "judge_evaluation": judge_content,
    }
    json_file.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("JSON results saved to: %s", json_file.absolute())


if __name__ == "__main__":
    main()
