from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
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

# Schemas for structured outputs
class QueryAnalysis(BaseModel):
    """Analysis output from Query Understanding Agent"""
    query_type: Literal["clear", "needs_clarification", "emergency"] = Field(
        description="Whether the query is understood, needs clarification, or is an emergency"
    )
    domain: Domains = Field(
        description="Main domain of the query" # here some generic domain but once we know more we can add/refine
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
    topics: List[str] = [

    ]
    clarification_options: Optional[List[str]] = [
    # here we can add some of the multi-choice questions to help the user express their need
    # I live that empty because this needs some brainstorming 🧠⛈️🌪️
        "Can you confirm whether your question is related to any of the following: " 
    ]


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

    system_prompt = """You are an expert query analyzer for a Red Cross virtual assistant.
    Analyze the query to understand the user's needs, emotional state, and language.
    Pay special attention to any signs of emergency or urgent needs.

    If you detect any of these, mark as EMERGENCY:
    - Immediate danger
    - Medical emergencies
    - Severe distress
    - Threats to basic safety

    If you cannot clearly understand the query, generate 2-3 clarifying options.

    Important: You will analyze:
    1. Query clarity and type (clear/needs_clarification/emergency)
    2. Domain of need
    3. Emotional state from language and content
    4. Language of query
    5. Confidence in understanding
    6. Key entities (locations, dates, needs)
    """

    # Get structured analysis from LLM
    structured_analysis = llm.with_structured_output(QueryAnalysis).invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {state['query']}\nLocation: {state.get('location', 'Not provided')}"}
        ]
    )

    print(structured_analysis)

    if structured_analysis.query_type == "needs_clarification":
        # Create clarification message with options
        options = "\n".join(f"- {opt}" for opt in structured_analysis.clarification_options)
        return Command(
            goto="await_clarification",
            update={
                "messages": [
                    {"role": "assistant", "content": f"To better help you, could you clarify if you mean:\n{options}"}],
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


# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("query_understanding", query_understanding_node)
workflow.add_edge(START, "query_understanding")

# Compile
graph = workflow.compile()
