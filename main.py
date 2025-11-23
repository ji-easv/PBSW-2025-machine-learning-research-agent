import os
import dotenv
from tools import search_web, search_research_papers_api
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


def main():
    task = """Find a research paper on speed bumps that was published after 2003 and has over 10 citations.
    Return the top three articles with their titles, authors, publication years, number of citations, and URLs."""

    paper_result = user_proxy.initiate_chat(
        research_paper_api_assistant,
        message=f"Task: {task}",
        max_turns=10,
    )

    web_result = user_proxy.initiate_chat(
        web_search_assistant, message=f"Task: {task}", max_turns=10
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
    )


if __name__ == "__main__":
    main()
