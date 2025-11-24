from autogen import AssistantAgent

from utils.utils import get_llm_config


def get_internal_critic_agent(api_key: str) -> AssistantAgent:
    internal_critic = AssistantAgent(
        name="internal_critic",
        llm_config=get_llm_config(api_key),
        system_message=(
            f"""
        You are an internal critic reviewing an agent's drafts.
        You only ever see the USER_REQUEST and an agent's messages.

        Evaluation criteria:
        - relevance

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
