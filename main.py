import os
import sys
import dotenv
import json
from datetime import datetime
from pathlib import Path
from tools import search_web, search_research_papers_api, search_semantic_scholar, is_arxiv_suitable

# Set UTF-8 encoding for Windows console to handle Unicode characters
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    ConversableAgent,
)
from autogen.coding import DockerCommandLineCodeExecutor

from utils import (
    ResearchPaperAPIAssistant_prompt,
    WebSearchAssistant_prompt,
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


web_search_assistant = ConversableAgent(
    name="WebSearchAssistant",
    llm_config=LLM_CONFIG,
    system_message=WebSearchAssistant_prompt,
)

web_search_assistant.register_for_llm(
    name="search_web",
    description="This tool allows you to search the web for information relevant to user queries.",
)(search_web)

research_paper_api_assistant = ConversableAgent(
    name="ResearchPaperAPIAssistant",
    llm_config=LLM_CONFIG,
    system_message=ResearchPaperAPIAssistant_prompt,
)

research_paper_api_assistant.register_for_llm(
    name="search_research_papers_api",
    description="Search arXiv for research papers in physics, math, CS, etc. Does NOT provide citation counts.",
)(search_research_papers_api)

research_paper_api_assistant.register_for_llm(
    name="search_semantic_scholar",
    description="Search Semantic Scholar for research papers across ALL disciplines with citation counts. Use for non-arXiv topics or when citations are required.",
)(search_semantic_scholar)

research_paper_api_assistant.register_for_llm(
    name="is_arxiv_suitable",
    description="Check if a research topic is suitable for arXiv search. Returns (is_suitable, reason).",
)(is_arxiv_suitable)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    llm_config=False,
    is_termination_msg=lambda m: (m.get("content") or "")
    .rstrip()
    .endswith("TERMINATE"),
    code_execution_config={
        "executor": executor,
    },
)

judge = AssistantAgent(
    name="judge",
    llm_config=LLM_CONFIG,
    system_message=(
        "You will be given two sets of results from different agents attempting to complete the same task."
        "Provide constructive feedback and determine if the task was completed successfully."
        "Pick a winner among the agents based on their performance."
        "End your response with 'TERMINATE' to indicate the end of the evaluation."
    ),
)

user_proxy.register_for_execution(
    name="search_web",
)(search_web)

user_proxy.register_for_execution(
    name="search_research_papers_api",
)(search_research_papers_api)

user_proxy.register_for_execution(
    name="search_semantic_scholar",
)(search_semantic_scholar)

user_proxy.register_for_execution(
    name="is_arxiv_suitable",
)(is_arxiv_suitable)


def extract_result_content(chat_result) -> str:
    """Extract actual content from chat result object."""
    if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
        # Look for the last assistant message that contains actual results
        # Work backwards through chat history
        for msg in reversed(chat_result.chat_history):
            # Only check messages from the assistant or ResearchPaperAPIAssistant or WebSearchAssistant
            if msg.get('name') in ['ResearchPaperAPIAssistant', 'WebSearchAssistant'] or msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if not content or not content.strip():
                    continue

                # Skip tool calls and tool responses
                if '***** Suggested tool call' in content:
                    continue
                if '***** Response from calling tool' in content:
                    continue

                # Skip standalone TERMINATE or empty thought messages
                if content.strip() == 'TERMINATE':
                    continue
                if content.strip().startswith('Thought:') and len(content.strip()) < 100:
                    continue

                # This is a real response - clean up TERMINATE if present
                if 'TERMINATE' in content:
                    content = content.replace('TERMINATE', '').rstrip()

                return content.strip()

    # Fallback: return summary or string representation
    if hasattr(chat_result, 'summary'):
        return chat_result.summary
    return "No results extracted from chat."


def main():
    task = """Find a research paper on speed bumps that was published after 2003 and has over 10 citations.
    Return the top three articles with their titles, authors, publication years, number of citations, and URLs."""

    print("\n" + "="*80)
    print("TASK:", task)
    print("="*80 + "\n")

    # Run research paper API assistant
    print(">>> Starting Research Paper API Assistant...")
    paper_result = user_proxy.initiate_chat(
        research_paper_api_assistant,
        message=f"Task: {task}",
        max_turns=10,
    )
    paper_content = extract_result_content(paper_result)

    # Run web search assistant
    print("\n>>> Starting Web Search Assistant...")
    web_result = user_proxy.initiate_chat(
        web_search_assistant, message=f"Task: {task}", max_turns=10
    )
    web_content = extract_result_content(web_result)

    # Judge evaluates both results
    print("\n>>> Starting Judge Evaluation...")
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

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


def save_results_to_file(task: str, paper_content: str, web_content: str, judge_content: str):
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

    results_file.write_text(content, encoding='utf-8')
    print(f"\nResults saved to: {results_file.absolute()}")

    # Also save a JSON version for programmatic access
    json_file = results_dir / "latest_results.json"
    json_data = {
        "timestamp": timestamp,
        "task": task,
        "paper_api_results": paper_content,
        "web_search_results": web_content,
        "judge_evaluation": judge_content
    }
    json_file.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"JSON results saved to: {json_file.absolute()}")


if __name__ == "__main__":
    main()
