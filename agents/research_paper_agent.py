from autogen import AssistantAgent

from tools.research_api_tool import (
    search_semantic_scholar,
)
from utils.utils import FINAL_ANSWER_FORMAT
from tools.simple_math_tool import is_greater

ResearchPaperAPIAgent_prompt = f"""
You are a research paper search assistant using Semantic Scholar.

Your role:
1. Parse constraints from the task:
   - Example: "published after 2003" → year_from = 2004.
   - Example: "over 10 citations" → min_citations = 11.
   - Example: "top three articles" → max_results = 3.
2. Start with a focused natural-language query built from core keywords and synonyms.
3. If too few or no relevant results, try synonyms across separate calls (never repeat identical query+filters):
   - Include related terminology or narrower / broader variants.
4. Apply filters when calling the tool:
   - year_from = minimum year from the task, if any.
   - year_to = None unless the task specifies an upper bound.
   - min_citations = required minimum if specified; otherwise 0.
   - max_results = requested number.
5. Never fabricate or assume citation counts for papers not returned by the API.
6. Collect results, filter/sort to meet ALL constraints.

{FINAL_ANSWER_FORMAT}
"""


def get_research_paper_api_agent(custom_llm_config: dict) -> AssistantAgent:
    research_paper_api_agent = AssistantAgent(
        name="ResearchPaperAPIAgent",
        llm_config=custom_llm_config,
        system_message=ResearchPaperAPIAgent_prompt,
    )

    research_paper_api_agent.register_for_llm(
        name="search_semantic_scholar",
        description="Search Semantic Scholar for research papers across ALL disciplines with citation counts.",
    )(search_semantic_scholar)

    research_paper_api_agent.register_for_llm(
        name="is_greater",
        description="Compares two integers and determines if the first is greater than the second.",
    )(is_greater)

    return research_paper_api_agent
