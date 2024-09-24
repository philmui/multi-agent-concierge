###################################################################################################
# company_research_agent.py
# --------------------------------------------------------------------------------------------------
# An agent that looks up company and equity research.
#
# @ Phil Mui
# Tue Sep 24 00:07:14 PDT 2024
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("company_research_agent")

import asyncio
import pprint
from datetime import datetime
from typing import List
from llama_index.core.agent import AgentChatResponse
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

from atlas.actions.finance.finance_tools import FinanceResearchTool

class CompanyResearchAgent:
    def __init__(self, state: dict):
        self.state = state
        self.finance_research_tool = FinanceResearchTool()
        self.agent = self._create_agent()
        self.memory = self.agent.memory

    def _create_agent(self) -> OpenAIAgent:
        tools = [
            FunctionTool.from_defaults(fn=self.finance_research_tool.get_company_research, name="get_company_research", description="Get company research information for a given stock symbol."),
            FunctionTool.from_defaults(fn=self.finance_research_tool.get_equity_research, name="get_equity_research", description="Get equity research information for a given stock symbol."),
            FunctionTool.from_defaults(fn=self.finance_research_tool.get_latest_business_news, name="get_latest_business_news", description="Get key business news and latest company information for a given stock symbol."),
            FunctionTool.from_defaults(fn=self.finance_research_tool.get_company_leadership, name="get_company_leadership", description="Get company leadership information for a given stock symbol."),
            FunctionTool.from_defaults(fn=self.done, name="done", description="Call this tool when you have completed your company or equity research."),
        ]

        system_prompt = f"""
            You are a helpful assistant that is performing company or equity research, as well as looking up key business news and latest company information given a stock symbol.
            Your task is to perform comprehensive research on a company including details such as the company's name, industry, market cap, recent news and information, etc.
            To do this, you must first try to look up the stock symbol for the company.
            The current user state is:
            {pprint.pformat(self.state, indent=4)}
            The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
            
            When you have completed your research, call the tool "done" to signal that you are done.
            If the user asks to do anything other than company or equity research, or looking up key business news and latest company information, 
            call the tool "done" to signal some other agent should help.
        """

        return OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o"),
            system_prompt=system_prompt,
        )

    def done(self, research_result: str) -> None:
        """When you complete your company or equity research task, call this tool and pass in your final research output."""
        _logger.info(f"Company or equity research is complete")
        self.state["company_research"] = research_result
        self.state["current_agent"] = None
        self.state["just_finished"] = True

    def chat(self, message: str, chat_history: List[str] = None) -> AgentChatResponse:
        return self.agent.chat(message, chat_history=chat_history)

    async def achat(self, message: str, chat_history: List[str] = None) -> AgentChatResponse:
        return await asyncio.to_thread(self.chat, message, chat_history)

