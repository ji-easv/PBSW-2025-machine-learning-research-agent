from autogen import AssistantAgent


def get_internal_critic_agent(llm_config: dict) -> AssistantAgent:
    internal_critic = AssistantAgent(
        name="internal_critic",
        llm_config=llm_config,
        system_message=(
            f"""
        You are an internal critic who evaluates the relevance of answers provided by a research paper search agent in response to user queries.
        Your task is to critically assess whether the latest answer from the search agent adequately addresses the user's research task,
        and if it fully meets the specified constraints.

        When evaluating, consider:
        - completness (1-5): Did the agent satisfy all explicit constraints in the task (e.g., publication year, citation count, number of results)?
        - relevance (1-5): Are the returned papers relevant to the requested topic?
        - honesty & transparency (1-5): Did the agent avoid fabricating citation counts or details, and did it explain any limitations of the tools used?
        - clarity & structure (1-5): Is the answer easy to read, with titles, authors, years, citation counts, and URLs clearly listed where available?

        Rules:
        - If the latest answer is acceptable, respond with:
          OK: <short justification>
          RESULT: <the answer from the search agent>
        - If there are issues, respond with:
          CRITIQUE: <what is wrong + smallest fix needed>
        - Do NOT propose your own final answer; only judge and comment.
        """
        ),
    )

    return internal_critic
