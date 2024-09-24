###################################################################################################
# orchestrator_agent.py
# --------------------------------------------------------------------------------------------------
# An agent that orchestrates the flow of the conversation, selecting the next agent to run based on the
# current state and chat history.
#
# @ Phil Mui
# Tue Sep 24 00:07:14 PDT 2024
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("orchestrator_agent")


import asyncio
from datetime import datetime
import pprint
from typing import List

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

from .finance_config import AgentName

ORCHESTRATOR_PROMPT_TMPL = (f"""
You are an expert orchestration agent choosing which agent to run next to fully answer the user query.
Your job is to respond with the name of the agent to run next, based on the current state, chat history and the user query. 
Agents are identified by short string names.  You do not respond with anything else.

## The current state
{{state}}

## The current chat history
{{chat_history}}

## The current date
{ datetime.now().strftime("%Y-%m-%d") }

## Names of agents

If a current_agent is already selected in the state, simply output that value.

If there is no current_agent value, look at the chat history and the current state and you MUST return one of the strings inside the double-quotes 
identifying an agent to run (do not include any other text nor the double quotes):

* "{AgentName.STOCK_LOOKUP.value}" - if they user wants to look up a stock price or other equity information
* "{AgentName.COMPANY_RESEARCH.value}" - if they user wants to look up company or equity research
* "{AgentName.INDUSTRY_RESEARCH.value}" - if they user wants to look up industry, sector, or country research
* "{AgentName.CONSUMER_RESEARCH.value}" - if they user wants to look up consumer trends analysis and sentiment analysis on recent news articles related to a specific query.
* "{AgentName.CONCIERGE.value}" - if the user wants to do something else, or hasn't said what they want to do, or you can't figure out what they want to do. Choose this by default.

**Response format instruction**: Output ONLY one of the above names of agents, without quotes, and nothing else.
NEVER respond with anything other than one of the above five strings. DO NOT be helpful or conversational.

---------------
User query: {{user_query}}
---------------

Your response: """)

orchestrator_prompt:PromptTemplate = PromptTemplate(ORCHESTRATOR_PROMPT_TMPL)


class OrchestrationAgent:
    def __init__(self, state: dict):
        self.state = state
        self.agent = self._create_agent()

    def _create_agent(self) -> OpenAIAgent:
        tools = [
            FunctionTool.from_defaults(fn=self.has_ticker, name="has_ticker", description="Check if a ticker symbol is set."),
            FunctionTool.from_defaults(fn=self.has_company_research, name="has_company_research", 
                                       description="Check if company or equity research is complete."),
            FunctionTool.from_defaults(fn=self.has_industry_research, name="has_industry_research", 
                                       description="Check if industry, sector, or country research is complete."),
            FunctionTool.from_defaults(fn=self.has_consumer_research, name="has_consumer_research", 
                                       description="Check if consumer trends analysis and sentiment analysis on recent news articles related to a specific query is complete."),
        ]
        
        system_prompt = f"""\
You are an expert orchestration agent choosing which agent to run next.
Your job is to decide which agent to run based on the current state of the user and what they've asked to do. 
Agents are identified by short string names.  What you do is return the name of the agent to run next. You do not do anything else.

If a current_agent is already selected in the state, simply output that value."""

        return OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o", temperature=0.01),
            system_prompt=system_prompt,
            verbose=True,
            memory=ChatMemoryBuffer.from_defaults(token_limit=8192)
        )

    def has_ticker(self) -> bool:
        """Useful for checking if a company ticker is set."""
        ticker = self.state["ticker"]
        _logger.info(f"Orchestrator checking if a known ticker is set: {ticker}.")
        return (ticker is not None)

    def has_company_research(self) -> bool:
        """Useful for checking if a company or equity research is complete."""
        ticker = self.state["ticker"]
        research = self.state["company_research"]
        _logger.info(f"Orchestrator checking if company or equity research is complete for {ticker}.")
        return (ticker is not None and research is not None)
    
    def has_industry_research(self) -> bool:
        """Useful for checking if a industry research is complete."""
        research = self.state["industry_research"]
        _logger.info(f"Orchestrator checking if industry, sector, or country research is complete.")
        return (research is not None)

    def has_consumer_research(self) -> bool:
        """Useful for checking if a consumer research is complete."""
        research = self.state["consumer_research"]
        _logger.info(f"Orchestrator checking if consumer trends analysis and sentiment analysis on recent news articles related to a specific query is complete.")
        return (research is not None)

    def chat(self, message: str, chat_history: List[str] = None, state: dict = None) -> str:
        if state is not None:
            self.state = state
        return self.agent.chat(message, chat_history=chat_history)

    async def achat(self, message: str, chat_history: List[str] = None, state: dict = None) -> str:
        return await asyncio.to_thread(self.chat, message, chat_history)