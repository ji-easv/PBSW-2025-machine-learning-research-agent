import os
import dotenv
from tools import search_web, search_research_papers_api
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    ConversableAgent,
)
from autogen.coding import DockerCommandLineCodeExecutor

from utils import get_work_dir

dotenv.load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")


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

web_search_assistant = ConversableAgent(
    name="WebSearchAssistant",
    llm_config=LLM_CONFIG,
    system_message="You are an expert web search assistant. Use the web search tool to find relevant information.",
)

web_search_assistant.register_for_llm(
    name="search_web",
    description="This tool allows you to search the web for information relevant to user queries.",
)(search_web)

research_paper_api_assistant = ConversableAgent(
    name="ResearchPaperAPIAssistant",
    llm_config=LLM_CONFIG,
    system_message="""You are an expert research paper search assistant specializing in arXiv API queries.

    When searching for papers:
    1. Use specific field prefixes for better results:
    - ti:"keyword" - search in titles
    - au:author_name - search by author
    - abs:"keyword" - search in abstracts
    - cat:category - filter by subject category (e.g., physics.class-ph, cs.AI)
    
    2. Use Boolean operators to refine searches:
    - AND - combine terms (e.g., ti:"traffic safety" AND cat:physics)
    - OR - alternative terms (e.g., ti:"speed bump" OR ti:"speed hump")
    - ANDNOT - exclude terms

    3. If you get no results or poor results:
    - Try broader search terms
    - Remove overly specific constraints
    - Try related keywords or synonyms
    - Search in different fields
    - Try related categories if category search is too narrow

    4. For citation or date requirements, note that arXiv API doesn't directly provide citation counts.
    You may need to inform the user of this limitation and provide the best available results.

    Example good queries:
    - ti:"machine learning" AND cat:cs.AI
    - au:hinton AND ti:neural
    - abs:"deep learning" OR abs:"neural network"
    - ti:"speed bump" OR ti:"road safety"

    Always start with a focused query, then broaden if needed.""",
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

judge = AssistantAgent(
    name="judge",
    llm_config=LLM_CONFIG,
    system_message=(
        "You will be given two sets of results from different agents attempting to complete the same task."
        " Provide constructive feedback and determine if the task was completed successfully."
        " Pick a winner among the agents based on their performance."
    ),
)

user_proxy.register_for_execution(
    name="search_web",
)(search_web)

user_proxy.register_for_execution(
    name="search_research_papers_api",
)(search_research_papers_api)


def main():
    task = """Find a research paper on speed bumps that was published after 2003 and has 10 citations.
    Return the top three articles with their titles, authors, publication years, number of citations, and URLs."""

    paper_result = user_proxy.initiate_chat(
        research_paper_api_assistant,
        message=f"Task: {task}",
        max_turns=5,
        clear_history=True,
    )

    web_result = user_proxy.initiate_chat(
        web_search_assistant,
        message=f"Task: {task}",
        max_turns=5,
        clear_history=True,
    )

    # Step 3: Judge evaluates both results
    judge_result = user_proxy.initiate_chat(
        judge,
        message=f"""Evaluate the performance of the research agents:

                Task: {task}

                Research Paper Results:
                {paper_result}

                Web Search Results:
                {web_result}

                Provide your evaluation and pick the best results.""",
        max_turns=2,
        clear_history=True,
    )


if __name__ == "__main__":
    main()
