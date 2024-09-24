###################################################################################################
# quality_eval_agent.py
# --------------------------------------------------------------------------------------------------
# An agent that evaluates the quality of the response so far and either asks for more information,
# or if the response is satisfactory, it will move on to the next agent.
#
# @ Phil Mui
# Tue Sep 24 00:07:14 PDT 2024
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("quality_eval_agent")


import asyncio
from datetime import datetime
import pprint
from typing import List
from llama_index.core.agent import AgentChatResponse
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

from llama_index.core.prompts import PromptTemplate

QUALITY_EVALUATION_PROMPT_TMPL = """
You are a quality evaluation agent responsible for determining if the original user query has been satisfactorily answered.

1. Review the chat history, focusing on the original user query and subsequent responses.
2. Assess the quality and completeness of the information provided in response to the original query.
3. Determine if further information or clarification is needed to fully address the user's request.

If the original query has been satisfactorily answered:
- Respond with "no_further_task" and nothing else.

If the answer quality is low or incomplete:
- Formulate a follow-up question or request that would help address the gaps in the current response.
- Your response should be phrased from the user's perspective, as if they were asking for more information.

Remember: Focus on the original user query and ensure that all aspects of their request have been adequately addressed.
"""

quality_evaluation_prompt:PromptTemplate = PromptTemplate(QUALITY_EVALUATION_PROMPT_TMPL)

class QualityEvalAgent:
    SYSTEM_PROMPT_TEMPLATE = """
    You are an expert in determining if your response is sufficient in answering the user's original question. 
    They might have had to complete some sub-tasks as part of that original task.  
    If you think the user's original task is complete, respond with "no_further_task".
    If you think the user's original task is not complete, re-emphasize the original question.
    
    The current state of the user is:
    {state}
    """

    def __init__(self, state: dict):
        self.state = state
        self.system_prompt = PromptTemplate(self.SYSTEM_PROMPT_TEMPLATE)
        self.agent = self._create_agent()

    def _create_agent(self) -> OpenAIAgent:
        def dummy_tool() -> bool:
            """A tool that does nothing."""
            _logger.info(f"Doing nothing.")

        tools = [
            FunctionTool.from_defaults(fn=dummy_tool)
        ]

        formatted_system_prompt = self.system_prompt.format(
            state=pprint.pformat(self.state, indent=4)
        )

        return OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o", temperature=0.4),
            system_prompt=formatted_system_prompt,
        )

    def chat(self, message: str, chat_history: List[str] = None) -> AgentChatResponse:
        return self.agent.chat(message, chat_history=chat_history)

    async def achat(self, message: str, chat_history: List[str] = None) -> AgentChatResponse:
        return await asyncio.to_thread(self.chat, message, chat_history)



