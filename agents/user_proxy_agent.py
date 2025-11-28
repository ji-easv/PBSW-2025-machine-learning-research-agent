from autogen import UserProxyAgent

from tools.research_api_tool import (
    search_semantic_scholar,
)
from tools.web_search_tool import check_pages_for_relevance, search_web
from autogen.coding import DockerCommandLineCodeExecutor


def get_user_proxy(executor: DockerCommandLineCodeExecutor) -> UserProxyAgent:
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        llm_config=False,
        is_termination_msg=lambda msg: (
            isinstance(msg, dict)
            and isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
        code_execution_config={
            "executor": executor,
        },
    )

    user_proxy.register_for_execution(
        name="search_web",
    )(search_web)

    user_proxy.register_for_execution(
        name="search_semantic_scholar",
    )(search_semantic_scholar)

    user_proxy.register_for_execution(
        name="check_pages_for_relevance",
    )(check_pages_for_relevance)

    return user_proxy
