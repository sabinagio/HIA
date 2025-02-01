import typing
from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict, Union
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.types import Command
from src.utils.llm_utils import get_api_key
import os

api_key = get_api_key()

Domains = Literal[
    "Where to go first",
    "Shelter",
    "Health & Wellbeing",
    "Dentist",
    "Safety & Protection",
    "Food & Clothing",
    "Work",
    "Asylum & Return",
    "Legal Advice",
    "Search Missing Relatives",
    "Women",
    "Children & Youth",
    "Courses & Activities",
    "Feedback",
    "Helpdesk & Social Support",
]

DomainWithOther = Union[Domains, Literal["Other"]]

# Schemas for structured outputs
class QueryAnalysis(BaseModel):
    """Analysis output from Query Understanding Agent"""
    query_type: Literal["clear", "needs_clarification", "emergency"] = Field(
        description="Whether the query is understood, needs clarification, or is an emergency"
    )
    domains: List[DomainWithOther] = Field(
        description="List of relevant domains for the query",
        default_factory=list
    )
    emotional_state: str = Field(
        description="Detected emotional state of the user"
    )
    language: str = Field(
        description="Detected language of the query"
    )
    confidence: float = Field(
        description="Confidence level in understanding (0-1)"
    )
    extracted_entities: dict = Field(
        description="Key entities from query (locations, dates, specific needs)"
    )
    topics: List[str] = []
    clarification_options: List[str] = Field(
        default_factory=lambda: list(typing.get_args(Domains))
    )
    print(f"clarification_options: {clarification_options}")
    # here we can add some of the multi-choice questions to help the user express their need
    # I live that empty because this needs some brainstorming 🧠⛈️🌪️

class AgentState(TypedDict):
    """State for Query Understanding Agent"""
    messages: Annotated[list, add_messages]
    query: str
    location: Optional[str]
    analysis: Optional[QueryAnalysis]


def query_understanding_node(state: AgentState):
    """
    Analyzes user query and routes to appropriate next steps.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022", # cheapest claude model
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    domain_list = "\n    - ".join(typing.get_args(Domains))

    system_prompt = """You are an expert query analyzer for a Red Cross virtual assistant.
    Analyze the query to understand the user's needs, emotional state, and language.
    Pay special attention to any signs of emergency or urgent needs.

    If you detect any of these, mark as EMERGENCY:
    - Immediate danger
    - Medical emergencies 
    - Severe distress
    - Threats to basic safety

    If the query is unclear, you should return relevant domains from this list as clarification options:
    {}

    Important: You will analyze:
    1. Query clarity and type (clear/needs_clarification/emergency)
    2. Domains of need, from this list of options {} or "Other"
    3. Emotional state from language and content
    4. Language of query
    5. Confidence in understanding
    6. Key entities (locations, dates, needs)
    """.format(domain_list, domain_list)

    # Get structured analysis from LLM
    structured_analysis = llm.with_structured_output(QueryAnalysis).invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {state['query']}\nLocation: {state.get('location', 'Not provided')}"}
        ]
    )

    print(structured_analysis)

    if structured_analysis.query_type == "needs_clarification":
        # Filter clarification options to only include valid domains
        structured_analysis.clarification_options = list(typing.get_args(Domains))
        # Create clarification message with domain options
        options = "\n".join(f"- {opt}" for opt in structured_analysis.clarification_options)
        return Command(
            goto="await_clarification",
            update={
                "messages": [
                    {"role": "assistant", "content": f"To better help you, please select the area(s) where you need assistance:\n{options}"}
                ],
                "analysis": structured_analysis.model_dump()
            }
        )

    elif structured_analysis.query_type == "emergency":
        # Route to emergency response
        return Command(
            goto="emergency_response",
            update={
                "messages": state["messages"],
                "analysis": structured_analysis.model_dump()
            }
        )

    else:
        # Query is clear - prepare outputs for Context Management and RAG agents
        context_update = {
            "user_context": {
                "language": structured_analysis.language,
                "emotional_state": structured_analysis.emotional_state,
                "domain": structured_analysis.domain,
                "location": state.get("location"),
                "entities": structured_analysis.extracted_entities
            }
        }

        rag_update = {
            "query_context": {
                "original_query": state["query"],
                "domain": structured_analysis.domain,
                "entities": structured_analysis.extracted_entities,
                "language": structured_analysis.language
            }
        }

        # Return updates for both agents
        # remove context management later - check if it's a good way to avoid hosting messages
        return [
            Command(
                goto="context_management",
                update=context_update
            ),
            Command(
                goto="rag",
                update=rag_update
            )
        ]
