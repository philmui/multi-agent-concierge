###################################################################################################
# industry_research_agent.py
# --------------------------------------------------------------------------------------------------
# An agent that looks up industry, sector, or country research.
#
# @ Phil Mui
# Tue Sep 24 00:07:14 PDT 2024
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("industry_research_agent")

import asyncio
from datetime import datetime
import pprint
from typing import List
from llama_index.core.agent import AgentChatResponse
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

from atlas.actions.finance.finance_tools import FinanceResearchTool

class IndustryResearchAgent:
    def __init__(self, state: dict):
        self.state = state
        self.agent = self._create_agent()
        self.memory = self.agent.memory

    def _create_agent(self) -> OpenAIAgent:
        finance_research_tool = FinanceResearchTool()

        def done(research_result: str) -> None:
            """When you have completed your industry, sector, or country research task, call this tool and pass in your final research output."""
            _logger.info(f"Industry, sector, region or country research is complete")
            self.state["industry_research"] = research_result
            self.state["current_agent"] = None
            self.state["just_finished"] = True

        tools = [
            FunctionTool.from_defaults(fn=finance_research_tool.get_industry_sector_research, name="get_industry_sector_research", description="Get industry or sector research information for a given industry or sector."),
            FunctionTool.from_defaults(fn=finance_research_tool.get_country_research, name="get_region_country_research", description="Get economic and financial data for a given country."),
            FunctionTool.from_defaults(fn=done, name="done", description="Call this tool when you have completed your industry, sector, or country research."),
        ]

        system_prompt = f"""
            You are a helpful assistant that is performing industry, sector or country research.
            Your task is to perform comprehensive research about an industry such as peer companies, sector trends, etc., or have retrieved data for a country such as GDP, inflation, etc.
            To do this, you must first try clarify which company, equity, industry, sector, or country the user is asking about.  
            Ask the user to clarify if they didn't specify enough information for you to do your research.
            The current user state is:
            {pprint.pformat(self.state, indent=4)}
            The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
            
            When you have completed your research, call the tool "done" to signal that you are done.
            If the user asks to do anything other than industry, sector, or country research, call the tool "done" to signal some other agent should help.
        """

        return OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o"),
            system_prompt=system_prompt,
        )

    def chat(self, message: str, chat_history: List[str] = None) -> str:
        return self.agent.chat(message, chat_history=chat_history)

    async def achat(self, message: str, chat_history: List[str] = None) -> str:
        return await asyncio.to_thread(self.chat, message, chat_history)

