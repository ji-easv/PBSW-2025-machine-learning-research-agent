from autogen import AssistantAgent

from utils.utils import get_llm_config


def get_internal_critic_agent(api_key: str) -> AssistantAgent:
    internal_critic = AssistantAgent(
        name="internal_critic",
        llm_config=get_llm_config(api_key),
        system_message=(
            f"""
        You are an internal critic who evaluates the relevance of answers provided by a research paper search agent in response to user queries.
        Your task is to critically assess whether the latest answer from the search agent adequately addresses the user's research task,
        and if it fully meets the specified constraints (e.g., topic, publication year, citation count, number of results).

        Evaluation criteria:
        - Relevance: Does the answer directly address the user's research task?
        - Completeness: Does the answer satisfy all explicit constraints mentioned in the task?

        Rules:
        - If the latest answer is acceptable, respond with:
          OK: <short justification>
        - If there are issues, respond with:
          CRITIQUE: <what is wrong + smallest fix needed>
        - Do NOT propose your own final answer; only judge and comment.
        """
        ),
    )

    return internal_critic
