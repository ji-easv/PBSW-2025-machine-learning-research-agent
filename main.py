import os
import sys
import dotenv
import logging
import argparse
from agents.external_judge_agent import (
    get_external_judge_agent,
    llm_judge_score,
)
from agents.research_paper_agent import get_research_paper_api_agent
from agents.search_orchestrator import SearchOrchestrator
from agents.user_proxy_agent import get_user_proxy
from agents.web_search_agent import get_web_search_agent
from utils.task_prompts import simple_tasks
from autogen.coding import DockerCommandLineCodeExecutor
from autogen import (
    GroupChat,
    GroupChatManager,
)
from utils.utils import (
    extract_final_answer,
    save_results,
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
parser = argparse.ArgumentParser(description="Research Paper Agent")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--llm-provider",
    choices=["mistral", "google", "cerebras"],
    dest="llm_provider",
    default="mistral",
)
group.add_argument(
    "--google",
    action="store_const",
    const="google",
    dest="llm_provider",
    help="Use Google as LLM provider",
)
group.add_argument(
    "--mistral",
    action="store_const",
    const="mistral",
    dest="llm_provider",
    help="Use Mistral as LLM provider",
)
group.add_argument(
    "--cerebras",
    action="store_const",
    const="cerebras",
    dest="llm_provider",
    help="Use Cerebras as LLM provider",
)

args = parser.parse_args()
llm_provider = args.llm_provider

if llm_provider == "mistral":
    env_var_name = "MISTRAL_API_KEY"
elif llm_provider == "google":
    env_var_name = "GOOGLE_API_KEY"
elif llm_provider == "cerebras":
    env_var_name = "CEREBRAS_API_KEY"
else:
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")

api_key = os.getenv(env_var_name)
if not api_key:
    raise ValueError(f"{env_var_name} not found in environment variables.")

LLM_CONFIG = get_llm_config(llm_provider=llm_provider, api_key=api_key)

executor = DockerCommandLineCodeExecutor(
    work_dir=get_work_dir(),
)

research_paper_api_agent = SearchOrchestrator(
    name="ResearchPaperAPIAgent",
    custom_llm_config=LLM_CONFIG,
    search_agent=get_research_paper_api_agent(custom_llm_config=LLM_CONFIG),
    executor=executor,
    human_input_mode="NEVER",
    llm_config=False,
)

web_search_agent = SearchOrchestrator(
    name="WebSearchOrchestrator",
    custom_llm_config=LLM_CONFIG,
    search_agent=get_web_search_agent(custom_llm_config=LLM_CONFIG),
    executor=executor,
    human_input_mode="NEVER",
    llm_config=False,
    terminate_conversation=True,
    max_turns=10,
)


user_proxy = get_user_proxy(executor=executor)
judge = get_external_judge_agent(custom_llm_config=LLM_CONFIG)


def has_result(agent_name: str, messages: list[dict]) -> bool:
    return any(
        msg.get("name") == agent_name and "RESULT:" in (msg.get("content") or "")
        for msg in messages
    )


def speaker_selection(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1] if messages else {}
    last_message_content = last_message.get("content", "") if messages else ""

    # kick-off the conversation
    if last_speaker is user_proxy and last_message_content.strip().startswith("TASK:"):
        return web_search_agent

    # research_paper_api_agent <-> user_proxy until RESULTS
    if not has_result("ResearchPaperAPIAgent", messages):
        if last_speaker is user_proxy:
            return research_paper_api_agent
        else:
            return user_proxy

    # web_search_agent <-> user_proxy until RESULTS
    if not has_result("WebSearchOrchestrator", messages):
        if last_speaker is user_proxy:
            return web_search_agent
        else:
            return user_proxy

    # once both RESULTS are in, end the conversation by injecting TERMINATE
    return user_proxy


def main():
    def should_terminate(msg) -> bool:
        if has_result("ResearchPaperAPIAgent", group.messages) and has_result(
            "WebSearchOrchestrator", group.messages
        ):
            return True
        return False

    group = GroupChat(
        agents=[user_proxy, research_paper_api_agent, web_search_agent],
        max_round=12,
        speaker_selection_method=speaker_selection,
    )

    manager = GroupChatManager(
        name="main_manager",
        groupchat=group,
        llm_config=LLM_CONFIG,
        is_termination_msg=should_terminate,
    )

    scores = []
    for task in simple_tasks:
        chat = user_proxy.initiate_chat(
            manager,
            message=f"TASK: {task}",
            max_turns=20,
            summary_method="reflection_with_llm",
        )

        research_paper_api_agent_result = extract_final_answer(
            chat, "ResearchPaperAPIAgent"
        )
        web_search_agent_result = extract_final_answer(chat, "WebSearchOrchestrator")

        judge_scores = llm_judge_score(
            judge,
            user_prompt=task,
            results={
                "ResearchPaperAPIAgent": research_paper_api_agent_result,
                "WebSearchOrchestrator": web_search_agent_result,
            },
        )
        logging.info(f"Judge Scores: {judge_scores}")
        scores.append(
            {
                "task": task,
                "ResearchPaperAPIAgent_result": research_paper_api_agent_result,
                "WebSearchOrchestrator_result": web_search_agent_result,
                "judge_scores": judge_scores,
            }
        )

    save_results(scores)


if __name__ == "__main__":
    main()
