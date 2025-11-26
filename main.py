import os
import sys
import dotenv
import json
import logging
from datetime import datetime
from pathlib import Path
from agents.internal_critic_agent import get_internal_critic_agent
from agents.research_paper_agent import get_research_paper_api_agent
from agents.search_orchestrator import SearchOrchestrator
from agents.user_proxy_agent import get_user_proxy
from agents.web_search_agent import get_web_search_agent
from autogen.coding import DockerCommandLineCodeExecutor
from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
)
from tools.web_search_tool import search_web
from utils.utils import (
    get_llm_config,
    get_work_dir,
)

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO
)

# Set UTF-8 encoding for Windows console to handle Unicode characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


dotenv.load_dotenv()
# api_key = os.getenv("MISTRAL_API_KEY")
# if not api_key:
#     raise ValueError("MISTRAL_API_KEY not found in environment variables.")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

LLM_CONFIG = get_llm_config(api_key=api_key)

task = """
Find research papers on software testing that satisfy ALL of the following constraints:
1) Published in 2024.
2) Have more than 10 citations.
3) Return the top three articles, providing for each: title, authors, publication year, number of citations, and URL.
"""

executor = DockerCommandLineCodeExecutor(
    work_dir=get_work_dir(),
)

web_search_agent = SearchOrchestrator(
    name="WebSearchOrchestrator",
    api_key=api_key,
    search_agent=get_web_search_agent(api_key=api_key),
    executor=executor,
    human_input_mode="NEVER",
    llm_config=False,
)

research_paper_api_agent = SearchOrchestrator(
    name="ResearchPaperAPIAgent",
    api_key=api_key,
    search_agent=get_research_paper_api_agent(api_key=api_key),
    executor=executor,
    human_input_mode="NEVER",
    llm_config=False,
)

user_proxy = get_user_proxy(executor=executor)

judge = AssistantAgent(
    name="judge",
    llm_config=LLM_CONFIG,
    system_message="""
        You are an external evaluator of research paper agents. You will be given two sets of results from different agents attempting to complete the same research task.
        Your role is to score and compare the quality of the results based on how well they meet the task requirements.
        
        When evaluating, consider:
        - completness (1-5): Did the agent satisfy all explicit constraints in the task (e.g., publication year, citation count, number of results)?
        - relevance (1-5): Are the returned papers relevant to the requested topic?
        - honesty & transparency (1-5): Did the agent avoid fabricating citation counts or details, and did it explain any limitations of the tools used?
        - clarity & structure (1-5): Is the answer easy to read, with titles, authors, years, citation counts, and URLs clearly listed where available?

        Return STRICT JSON, no extra commentary, end your response with TERMINATE.
    """,
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


def speaker_selection(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1] if messages else {}
    last_message_content = last_message.get("content", "") if messages else ""

    def has_result(agent_name):
        return any(
            msg.get("name") == agent_name and "RESULTS:" in (msg.get("content") or "")
            for msg in messages
        )

    # kick-off the conversation
    if last_speaker is user_proxy and last_message_content.strip().startswith("TASK:"):
        return research_paper_api_agent

    # research_paper_api_agent <-> user_proxy until RESULTS
    if not has_result("ResearchPaperAPIAgent"):
        if last_speaker is user_proxy:
            return research_paper_api_agent
        else:
            return user_proxy

    # web_search_agent <-> user_proxy until RESULTS
    if not has_result("WebSearchOrchestrator"):
        if last_speaker is user_proxy:
            return web_search_agent
        else:
            return user_proxy

    return judge


def main():
    group = GroupChat(
        agents=[user_proxy, web_search_agent, research_paper_api_agent, judge],
        max_round=12,
        speaker_selection_method=speaker_selection,
    )

    manager = GroupChatManager(
        name="main_manager",
        groupchat=group,
        llm_config=LLM_CONFIG,
    )

    user_proxy.initiate_chat(
        manager,
        message=f"TASK: {task}",
        max_turns=20,
        summary_method="reflection_with_llm",
    )


if __name__ == "__main__":
    main()
