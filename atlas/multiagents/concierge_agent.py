###################################################################################################
# concierge_agent.py
# --------------------------------------------------------------------------------------------------
# An agent that acts as a concierge for the user and coordinate across multiple agents.
#
# @ Phil Mui
# Tue Sep 24 00:07:14 PDT 2024
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("concierge_agent")

import asyncio
from datetime import datetime
import pprint
from typing import List
from llama_index.core.agent import AgentChatResponse
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

from llama_index.core.prompts import PromptTemplate

class ConciergeAgent:
    SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant that is helping a user perform business, company, and financial research.
Your job is to ask the user questions to figure out what they want to do, and give them the available things they can do.
That includes
* look up current and historical stock prices
* research on a company or equity research, as well as look up key business news and latest company information given a stock symbol.
* research on industry, sector, or country research
* understand consumer trends analysis and sentiment analysis on recent news articles related to a specific query.

The current state of the user is:
{state}
"""

    def __init__(self, state: dict):
        self.state = state
        self.agent = self._create_agent()

    def _create_agent(self) -> OpenAIAgent:
        def dummy_tool() -> bool:
            """A tool that does nothing."""
            _logger.info(f"Doing nothing.")

        tools = [
            FunctionTool.from_defaults(fn=dummy_tool)
        ]

        prompt_template = PromptTemplate(self.SYSTEM_PROMPT_TEMPLATE)
        system_prompt = prompt_template.format(state=pprint.pformat(self.state, indent=4))

        return OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o"),
            system_prompt=system_prompt,
        )

    def chat(self, message: str, chat_history: List[str] = None) -> str:
        return self.agent.chat(message, chat_history=chat_history)

    async def achat(self, message: str, chat_history: List[str] = None) -> str:
        return await asyncio.to_thread(self.chat, message, chat_history)

