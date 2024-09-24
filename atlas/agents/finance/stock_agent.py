###################################################################################################
# stock_agent.py
# --------------------------------------------------------------------------------------------------
# An agent that looks up current and historical stock prices.
#
# @ Phil Mui
# Tue Sep 24 00:07:14 PDT 2024
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("stock_agent")


import asyncio
from datetime import datetime
from enum import Enum
from typing import List
import pprint

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent

from atlas.actions.finance.finance_tools import FinanceResearchTool

# Stock lookup agent
class StockAgent:
    def __init__(self, state: dict):
        self.state = state
        self.agent = self._create_agent()
        self.memory = self.agent.memory

    def _create_agent(self) -> OpenAIAgent:
        def done(ticker: str) -> None:
            """When you have returned a stock price, call this tool."""
            _logger.info("Stock lookup is complete")
            self.state["ticker"] = ticker
            self.state["current_agent"] = None
            self.state["just_finished"] = True
        
        finance_research_tool = FinanceResearchTool()

        tools = [
            FunctionTool.from_defaults(fn=finance_research_tool.get_current_stock_price, name="lookup_current_stock_price", description="Look up the current stock price for a given stock symbol."),
            FunctionTool.from_defaults(fn=finance_research_tool.get_historical_stock_prices, name="lookup_historical_stock_prices", description="Look up historical stock prices for a given stock symbol."),
            FunctionTool.from_defaults(fn=finance_research_tool.search_for_stock_symbol, name="search_for_stock_symbol", description="Search for a stock symbol or basic equity information given the name of a company."),
            FunctionTool.from_defaults(fn=done, name="done", description="Call this tool with the ticker symbol when you are done looking up a stock price or other equity information."),
        ]

        system_prompt = (f"""
            You are a helpful assistant that is looking up stock prices -- both current and historical.
            The user may not know the stock symbol of the company they're interested in,
            so you can help them look it up by the name of the company.
            You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up. 
            Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.
            The current user state is:
            {pprint.pformat(self.state, indent=4)}
            The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
            
            Once you have supplied a stock price, you must call the tool "done" to signal that you are done.
            If the user asks to do anything other than look up a stock symbol or price, call the tool "done" to signal some other agent should help.
        """)

        return OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o"),
            system_prompt=system_prompt,
        )

    def chat(self, message: str, chat_history: List[str] = None) -> str:
        return self.agent.chat(message, chat_history=chat_history)

    async def achat(self, message: str, chat_history: List[str] = None) -> str:
        return await asyncio.to_thread(self.chat, message, chat_history)