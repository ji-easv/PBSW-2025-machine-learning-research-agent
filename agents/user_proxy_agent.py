from autogen import UserProxyAgent

from tools.fetch_link_tool import fetch_link
from tools.research_api_tool import (
    is_arxiv_suitable,
    search_research_papers_api,
    search_semantic_scholar,
)
from tools.web_search_tool import search_web
from autogen.coding import DockerCommandLineCodeExecutor


def get_user_proxy(executor: DockerCommandLineCodeExecutor) -> UserProxyAgent:
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

    user_proxy.register_for_execution(
        name="search_semantic_scholar",
    )(search_semantic_scholar)

    user_proxy.register_for_execution(
        name="is_arxiv_suitable",
    )(is_arxiv_suitable)

    # user_proxy.register_for_execution(
    #   name="fetch_link", description="Fetch a URL link."
    # )(fetch_link)

    return user_proxy
