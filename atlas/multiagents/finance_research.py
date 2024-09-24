###################################################################################################
# research.py
# --------------------------------------------------------------------------------------------------
# A multi-agent conversational system for analyzing a complex business analysis.
#
# Agent 1: look up current and historical stock prices.
# Agent 2: perform company and equity research.
# Agent 3: perform industry and sector research.
# Agent 4: perform consumer trends analysis andsentiment analysis on recent news articles related to a specific query.
# Agent 5: lookup key business news and latest company details given a stock symbol.
# Concierge agent: a catch-all agent that helps navigate between the other 4.
# Orchestration agent: decides which agent to run based on the current state of the user.
#
# Example query to try:
# > How is the US economy the past year?  Does it impact the stock price for Salesforce?  What about the general consumer sentiment?
#
# @ Phil Mui
# 
###################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("finance_research")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


import asyncio
from datetime import datetime
from enum import Enum
from typing import List
import pprint
from colorama import Fore, Back, Style

from llama_index.core.agent import AgentChatResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent

from atlas.actions.finance.finance_tools import FinanceResearchTool
from atlas.agents.finance import (
    ConsumerResearchAgent,
    CompanyResearchAgent, 
    IndustryResearchAgent,
    StockAgent, 
)

from atlas.multiagents import (
    ConciergeAgent,
    OrchestrationAgent,
    QualityEvalAgent,
    orchestrator_prompt,
    quality_evaluation_prompt,
)

from .finance_config import AgentName
    
def stock_lookup_agent_factory(state: dict) -> StockAgent:
    return StockAgent(state)

def company_research_agent_factory(state: dict) -> CompanyResearchAgent:
    return CompanyResearchAgent(state)

def industry_research_agent_factory(state: dict) -> IndustryResearchAgent:
    return IndustryResearchAgent(state)

def consumer_research_agent_factory(state: dict) -> ConsumerResearchAgent:
    return ConsumerResearchAgent(state)

def concierge_agent_factory(state: dict) -> ConciergeAgent:
    return ConciergeAgent(state)

def quality_evaluation_agent_factory(state: dict) -> QualityEvalAgent:
    return QualityEvalAgent(state)

def orchestration_agent_factory(state: dict) -> OrchestrationAgent:
    return OrchestrationAgent(state)


def get_initial_state() -> dict:
    return {
        "ticker": None,
        "company_research": None,
        "industry_research": None,
        "consumer_research": None,
        "current_agent": None,
        "just_finished": False,
    }

def run() -> None:
    state = get_initial_state()

    orchestration_agent = orchestration_agent_factory(state)
    root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

    first_run = True
    is_retry = False
    num_tries = 0

    while True:
        num_tries += 1
        if first_run:
            # if this is the first run, start the conversation
            user_msg_str = "Hello"
            first_run = False            
        elif is_retry == True:
            user_msg_str = "That's not right, try again. Pick one agent."
            is_retry = False
        elif state["just_finished"] == True and num_tries < 5:
            _logger.info("Asking the quality evaluation agent to decide what to do next")
            response:AgentChatResponse = quality_evaluation_agent_factory(state).chat(
                quality_evaluation_prompt.format(),
                chat_history=current_history
            )
            user_msg_str = str(response.response).strip()
            _logger.info(f"Quality evaluation agent said: {user_msg_str}")
            if "no_further_task" in user_msg_str:
                user_msg_str = input(">> ").strip()
                state["just_finished"] = False
                num_tries = 0             
        else:
            # any other time, get user input
            user_msg_str = input("> ").strip()            
            
            # reset the state for a new conversation
            state = get_initial_state()
            num_tries = 0             

        current_history = root_memory.get()
        
        # who should speak next?
        if (state["current_agent"]):
            _logger.info(f"There's already a speaker: {state['current_agent']}")
            next_speaker = state["current_agent"]
        else:
            _logger.info("No current agent selected, asking orchestration agent to decide")
            orchestration_response = orchestration_agent.chat(
                message=orchestrator_prompt.format(
                    state=state, 
                    chat_history=current_history, 
                    user_query=user_msg_str),
                state=state)
            next_speaker = str(orchestration_response).strip()

        #_logger.info(f"Next speaker: {next_speaker}")

        if next_speaker == AgentName.STOCK_LOOKUP:
            _logger.info("Stock lookup agent selected")
            current_agent = stock_lookup_agent_factory(state)
            state["current_agent"] = next_speaker
        elif next_speaker == AgentName.COMPANY_RESEARCH:
            _logger.info("Company research agent selected")
            current_agent = company_research_agent_factory(state)
            state["current_agent"] = next_speaker
        elif next_speaker == AgentName.INDUSTRY_RESEARCH:
            _logger.info("Industry research agent selected")
            current_agent = industry_research_agent_factory(state)
            state["current_agent"] = next_speaker
        elif next_speaker == AgentName.CONSUMER_RESEARCH:
            _logger.info("Consumer research agent selected")
            current_agent = consumer_research_agent_factory(state)
            state["current_agent"] = next_speaker
        elif next_speaker == AgentName.CONCIERGE:
            _logger.info("Concierge agent selected")
            current_agent = concierge_agent_factory(state)
            state["current_agent"] = next_speaker
        else:
            _logger.info("Orchestration agent failed to return a valid speaker; ask it to try again")
            is_retry = True
            continue

        pretty_state = pprint.pformat(state, indent=4)
        _logger.info(f"State: {pretty_state}")

        # chat with the current selected agent
        response = current_agent.chat(user_msg_str, chat_history=current_history)
        _logger.info(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # update chat history
        new_history = current_agent.memory.get_all()
        root_memory.set(new_history)

if __name__ == "__main__":
    run()
