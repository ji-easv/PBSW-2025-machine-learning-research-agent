import logging
from typing import Any
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.coding import DockerCommandLineCodeExecutor

from agents.internal_critic_agent import get_internal_critic_agent
from agents.user_proxy_agent import get_user_proxy
from utils.utils import MAX_INTERNAL_ROUNDS


class SearchOrchestrator(ConversableAgent):
    def __init__(
        self,
        api_key: str,
        search_agent: ConversableAgent,
        executor: DockerCommandLineCodeExecutor,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.user_proxy = get_user_proxy(executor)
        self.critic = get_internal_critic_agent(api_key)
        self.search_agent = search_agent
        self.group = GroupChat(
            agents=[self.search_agent, self.critic, self.user_proxy],
            max_round=MAX_INTERNAL_ROUNDS,
            speaker_selection_method=self.speaker_selection,
        )
        self.group_manager = GroupChatManager(
            name=self.name + "_group_manager", groupchat=self.group, llm_config=False
        )

    def speaker_selection(
        self, last_speaker, group: GroupChat
    ) -> ConversableAgent | None:
        messages = group.messages
        last_message_content = messages[-1].get("content", "") if messages else ""

        if last_speaker is self.user_proxy:
            if last_message_content.strip().startswith("TASK:"):
                return self.search_agent
            return self.critic
        elif last_speaker is self.search_agent:
            return self.user_proxy
        elif last_speaker is self.critic:
            if "OK:" in last_message_content:
                return None
            return self.search_agent
        return self.search_agent

    def generate_reply(
        self, messages=None, sender=None, exclude=None
    ) -> str | dict[str, Any] | None:
        if not messages:
            messages = []

        task = "No task provided."
        for msg in reversed(messages):
            content = msg.get("content", "")
            if content.strip().startswith("TASK:"):
                task = content
                break

        try:
            self.user_proxy.initiate_chat(
                self.group_manager,
                message=task,
                summary_method="last_msg",
                max_round=5,
            )
        except Exception as e:
            logging.error(f"Error during inner conversation: {e}")
            return f"Error during inner conversation. {e}"

        # Get the final winning solution from the judge
        final_messages = self.group_manager.groupchat.messages
        winning_solution = None

        for msg in reversed(final_messages):
            content = msg.get("content", "")
            if content.startswith("OK:"):
                winning_solution = content
                break

        return winning_solution or "No winning solution found."
