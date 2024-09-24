from enum import Enum

class AgentName(str, Enum):
    STOCK_LOOKUP = "stock_lookup"
    COMPANY_RESEARCH = "company_research"
    INDUSTRY_RESEARCH = "industry_research"
    CONSUMER_RESEARCH = "consumer_research"
    CONCIERGE = "concierge"
    ORCHESTRATOR = "orchestrator"
    