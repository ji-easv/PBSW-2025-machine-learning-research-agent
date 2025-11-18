from datetime import datetime
import os
from pathlib import Path
import dotenv
from tools import search_web, search_research_papers_api
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    ConversableAgent,
)
from autogen.coding import DockerCommandLineCodeExecutor

dotenv.load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")


def get_work_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    p = Path.cwd() / "coding" / timestamp
    p.mkdir(parents=True, exist_ok=True)
    return p


executor = DockerCommandLineCodeExecutor(
    work_dir=get_work_dir(),
)

LLM_CONFIG = {
    "config_list": [
        {
            "model": "mistral-small-2503",
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
            "timeout": 120,
        }
    ]
}

web_search_assistant = AssistantAgent(
    name="WebSearchAssistant",
    llm_config=LLM_CONFIG,
    system_message="You are an expert web search assistant. Use the web search tool to find relevant information.",
)

web_search_assistant.register_for_llm(
    name="search_web",
    description="This tool allows you to search the web for information relevant to user queries.",
)(search_web)

research_paper_api_assistant = AssistantAgent(
    name="ResearchPaperAPIAssistant",
    llm_config=LLM_CONFIG,
    system_message="You are an expert research paper search assistant. Use the research paper search tool to find relevant academic papers.",
)

research_paper_api_assistant.register_for_llm(
    name="search_research_papers_api",
    description="Search for academic research papers based on topic, year, and citation criteria.",
)(search_research_papers_api)

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

user_proxy.register_for_execution(
    name="search_web",
)(search_web)

user_proxy.register_for_execution(
    name="search_research_papers_api",
)(search_research_papers_api)

# group = GroupChat(
#     agents=[
#         web_search_assistant,
#         research_paper_api_assistant,
#         user_proxy,
#     ],
#     messages=[],
#     max_round=30,
#     speaker_selection_method="auto",
# )
# manager = GroupChatManager(groupchat=group, llm_config=LLM_CONFIG)

user_proxy.initiate_chat(
    web_search_assistant,
    message="Find a research paper on speed bumps that was published after 2003 and has 10 citations.",
    summary_method="reflection_with_llm",
)
